"""LangGraph workflow definition for Reinforcement Learning."""

from typing import Dict, Any
import logging

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph.state import CompiledStateGraph

from agent.core.state import FeedbackState


logger = logging.getLogger(__name__)


def create_rl_workflow(agent) -> CompiledStateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(FeedbackState)

    # Async wrappers for async nodes
    async def feedback_handling_node(state): return await _feedback_handling_node(state, agent)
    async def create_art_trajectory_node(state): return await _create_art_trajectory_node(state, agent)
    async def fine_tune_node(state): return await _fine_tune_check_node(state, agent)

    # Add nodes
    workflow.add_node("feedback_handling", feedback_handling_node)
    workflow.add_node("create_art_trajectory", create_art_trajectory_node)
    workflow.add_node("fine_tune", fine_tune_node)

    # Set entry point
    workflow.set_entry_point("feedback_handling")

    # Add edges
    workflow.add_conditional_edges(
        "feedback_handling",
        lambda state: state.get("is_feedback_valid", False),
        {
            True: "create_art_trajectory",
            False: END,
        }
    )

    workflow.add_edge(
        "create_art_trajectory",
        "fine_tune"
    )

    workflow.add_edge(
        "fine_tune",
        END
    )
    return workflow.compile()


async def _feedback_handling_node(state: FeedbackState, agent) -> Dict[str, Any]:
    """Handle user feedback collection and processing"""
    feedback_entry = state.get("feedback")
    if not feedback_entry:
        return {"is_feedback_valid": False}

    # Convert dataclass to dict for the validator
    feedback_dict = {
        "answer": feedback_entry.answer,
        "comment": feedback_entry.comment,
        "reaction": feedback_entry.rating  # Map rating to reaction
    }

    is_valid = agent.feedback_validator.is_valid(feedback_dict)

    logger.info(f"Feedback validity check: {'VALID' if is_valid else 'INVALID'}")

    return {"is_feedback_valid": is_valid}

async def _create_art_trajectory_node(state: FeedbackState, agent) -> Dict[str, Any]:
    """Create ART (Automatic Reward Training) trajectory for RL"""
    feedback_entry = state["feedback"]
    trainer = agent.rag_trainer
    trainer.add_feedback(feedback_entry.__dict__)

    logger.info("Created ART Trajectory from valid feedback.")
    return {}

async def _fine_tune_node(state: FeedbackState, agent) -> Dict[str, Any]:
    trainer = agent.rag_trainer
    if len(trainer.trajectories) >= 30:
        logger.info("Sufficient feedback collected. Starting fine-tuning.")
        trainer.train_on_feedback()
        trainer.trajectories.clear()
        return {"result": True}
    logger.info("Still insufficient feedback. Keep collecting.")
    return {"result": False}
