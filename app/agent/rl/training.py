"""Main Reinforcement Learning Trainer with ART Trajectories module"""

import json
import pandas as pd
import torch
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Unsloth imports for efficient fine-tuning
from unsloth import FastLanguageModel

# Additional imports for training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# Custom modules
from agent.rl import ARTTrajectory, FeedbackEntry


class RAGReinforcementTrainer:
    """
    Main class for implementing Reinforcement Learning on RAG Agent
    Uses Unsloth for efficient training and ART for trajectory management
    """

    def __init__(self,
                 model_path: str = "google/gemma-3-270m-it",
                 max_seq_length: int = 2048):

        self.model_path = Path(model_path)
        self.max_seq_length = max_seq_length

        # Initialize model and tokenizer
        self.model, self.tokenizer = self._load_model()
        self.trajectories: List[ARTTrajectory] = []

        # Training configuration
        self.training_config = {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 1,
            "warmup_steps": 10,
            "logging_steps": 1,
        }

    def _load_model(self):
        """Load model using Unsloth for efficient training"""
        print(f"Loading model from {self.model_path}")

        # Load model with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization for efficiency
        )

        # Apply LoRA (Low-Rank Adaptation) for efficient fine-tuning
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        return model, tokenizer

    def add_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """
        Add new feedback and create ART trajectory

        Args:
            feedback_data: Dictionary containing question, rating, answer, comment, correct_answer
        """
        feedback_entry = FeedbackEntry(**feedback_data)
        trajectory = ARTTrajectory(feedback_entry)
        self.trajectories.append(trajectory)

        # print(f"Added new feedback trajectory. Total trajectories: {len(self.trajectories)}")
        # print(f"Trajectory type: {trajectory.trajectory_data['training_type']}")
        # print(f"Reward signal: {trajectory.trajectory_data['reward']:.2f}")

    def load_feedback_dataset(self, dataset_path: str) -> None:
        """
        Load feedback dataset from CSV or JSON file
        Expected columns: question, rating, answer, comment, correct_answer
        """
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError("Dataset must be CSV or JSON format")

        print(f"Loading {len(df)} feedback entries from {dataset_path}")

        for _, row in df.iterrows():
            feedback_data = {
                "question": row["question"],
                "rating": float(row["rating"]),
                "answer": row["answer"],
                "comment": row["comment"],
                "correct_answer": row["correct_answer"]
            }
            self.add_feedback(feedback_data)

    def _create_training_dataset(self) -> List[Dict[str, str]]:
        """
        Convert ART trajectories into training format
        Creates different training examples based on trajectory type
        """
        training_data = []
        for trajectory in self.trajectories:
            traj_data = trajectory.trajectory_data
            training_type = traj_data["training_type"]

            if training_type == "positive_reinforcement":
                # Reinforce good responses
                training_example = {
                    "instruction": traj_data["query"],
                    "input": "",
                    "output": traj_data["response"]
                }
                training_data.append(training_example)

            elif training_type == "negative_correction":
                # Learn from corrections - use reference answer
                training_example = {
                    "instruction": traj_data["query"],
                    "input": f"Previous incorrect response: {traj_data['response']}\nFeedback: {traj_data['feedback_comment']}",
                    "output": traj_data["reference_answer"]
                }
                training_data.append(training_example)

            elif training_type == "neutral_refinement":
                # Blend current response with reference for improvement
                improved_response = self._create_improved_response(
                    traj_data["response"],
                    traj_data["reference_answer"],
                    traj_data["feedback_comment"]
                )
                training_example = {
                    "instruction": traj_data["query"],
                    "input": "",
                    "output": improved_response
                }
                training_data.append(training_example)

        return training_data

    def _create_improved_response(self, current: str, reference: str, feedback: str) -> str:
        """Create improved response by blending current and reference answers"""
        # Simple improvement strategy - you can make this more sophisticated
        if len(reference) > len(current):
            return reference  # Use reference if it's more comprehensive
        else:
            return f"{current} Additionally, {reference}"

    def _create_dataset(self, training_data: List[Dict[str, Dict[str, str]]]) -> List[str]:
        """Create dataset from training data"""
        formatted_data = []

        for example in training_data:
            # Create conversation format
            conversation = {
                "prompt": {"role": "user", "content": example["instruction"]},
                "completion": {"role": "assistant", "content": example["output"]}
            }

            formatted_data.append(conversation)
        return Dataset.from_list(formatted_data)

    def _format_for_chat_template(self, example: Dict[str, Dict[str, str]]) -> dict:
        """Format training data using chat template"""
        # Create conversation format
        conversation = [
            example["prompt"],
            example["completion"]
        ]

        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        tokenized = self.tokenizer(formatted, truncation=True, max_length=1024)
        return tokenized

    def train_on_feedback(self, output_dir: str = "improved_model") -> None:
        """
        Train model on accumulated feedback using ART trajectories
        """
        if len(self.trajectories) == 0:
            print("No feedback trajectories available. Add feedback first.")
            return

        print(f"Training on {len(self.trajectories)} feedback trajectories...")

        # Create training dataset
        training_data = self._create_training_dataset()
        formatted_data = self._create_dataset(training_data)
        preprocessed_data = formatted_data.map(self._format_for_chat_template)

        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.training_config["batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            warmup_steps=self.training_config["warmup_steps"],
            num_train_epochs=self.training_config["num_train_epochs"],
            learning_rate=self.training_config["learning_rate"],
            logging_steps=self.training_config["logging_steps"],
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            report_to="none",
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=preprocessed_data,
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        # Train the model
        print("Starting training...")
        trainer.train()

        # Save the model
        trainer.save_model()
        print(f"Model saved to {output_dir}")

    def evaluate_improvement(self, test_questions: List[str]) -> Dict[str, Any]:
        """
        Evaluate model improvement on test questions
        This is a simple evaluation - you can make it more sophisticated
        """
        print("Evaluating model improvement...")

        results = {
            "test_questions": test_questions,
            "responses": [],
            "evaluation_timestamp": datetime.now().isoformat()
        }

        for question in test_questions:
            # Generate response
            inputs = self.tokenizer(
                f"User: {question}\nAssistant:",
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            results["responses"].append({
                "question": question,
                "response": response.strip()
            })

        return results

    def save_trajectories(self, filepath: str) -> None:
        """Save ART trajectories for analysis"""
        trajectory_data = [traj.trajectory_data for traj in self.trajectories]

        with open(filepath, 'w') as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"Saved {len(trajectory_data)} trajectories to {filepath}")
