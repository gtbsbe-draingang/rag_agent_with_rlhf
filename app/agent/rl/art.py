"""Main Art class that gives trajectory control"""

import re
from typing import Dict, Any

import numpy as np
from agent.rl import FeedbackEntry


class ARTTrajectory:
    """
    ART (Automatic Response Training) Trajectory class
    Manages the creation and processing of training trajectories from feedback
    """

    def __init__(self, feedback_entry: FeedbackEntry) -> None:
        self.feedback = feedback_entry
        self.trajectory_data = self._create_trajectory()

    def _create_trajectory(self) -> Dict[str, Any]:
        """
        Create ART trajectory from feedback entry
        ART trajectories include the query, response, and reward signal

        Args:
            llm (Any): LLM to evaluate feedback
        """
        # Normalize rating to reward signal (-1 to 1)
        reward = self._calculate_reward(self.feedback)

        # Create preference data based on rating
        trajectory = {
            "query": self.feedback.question,
            "response": self.feedback.answer,
            "reward": reward,
            "reference_answer": self.feedback.correct_answer,
            "feedback_comment": self.feedback.comment,
            "timestamp": self.feedback.timestamp,
            "training_type": self._determine_training_type(reward)
        }

        return trajectory

    def _calculate_reward(
            self,
            feedback: FeedbackEntry,
            sentiment_weight: float = 0.3,
            length_weight: float = 0.1,
            specificity_weight: float = 0.2,
            normalize: bool = True
    ) -> float:
        """
        Calculate RLHF reward from rating (1-5) and feedback text.

        Args:
            feedback: Feedback Entry
            sentiment_weight: Weight for sentiment analysis component
            length_weight: Weight for feedback length component
            specificity_weight: Weight for feedback specificity component
            normalize: Whether to normalize final reward to [-1, 1] range

        Returns:
            Reward value
        """

        # Base reward from rating
        base_reward = 0.0
        if 1 < feedback.rating <= 5:
            base_reward = (feedback.rating - 3) / 2.0
        elif 0 <= feedback.rating <= 1:
            base_reward = feedback.rating

        # Analyze feedback text
        feedback_lower = feedback.comment.lower().strip()

        # Sentiment analysis (multilingual keyword-based)
        positive_keywords = [
            # English
            'good', 'great', 'excellent', 'amazing', 'helpful', 'useful',
            'clear', 'accurate', 'perfect', 'love', 'like', 'thank',
            'appreciate', 'wonderful', 'fantastic', 'brilliant', 'awesome',
            # Russian
            'хорошо', 'отлично', 'превосходно', 'замечательно', 'полезно',
            'помогает', 'ясно', 'понятно', 'точно', 'правильно', 'супер',
            'классно', 'круто', 'спасибо', 'благодарю', 'нравится',
            'люблю', 'великолепно', 'прекрасно', 'идеально', 'молодец',
            # Kazakh
            'жақсы', 'өте жақсы', 'тамаша', 'керемет', 'пайдалы',
            'көмектеседі', 'түсінікті', 'дұрыс', 'дәл', 'ұнайды',
            'рахмет', 'алғыс', 'сүйемін', 'ғажап', 'мықты'
        ]

        negative_keywords = [
            # English
            'bad', 'terrible', 'awful', 'wrong', 'useless', 'unhelpful',
            'confusing', 'unclear', 'hate', 'dislike', 'disappointed',
            'frustrating', 'annoying', 'stupid', 'horrible', 'waste',
            # Russian
            'плохо', 'ужасно', 'отвратительно', 'неправильно', 'бесполезно',
            'не помогает', 'непонятно', 'неясно', 'ненавижу', 'не нравится',
            'разочарован', 'расстроен', 'глупо', 'ерунда', 'чушь', 'ошибка',
            # Kazakh
            'жаман', 'нашар', 'қате', 'түсініксіз', 'пайдасыз',
            'көмектеспейді', 'ұнамайды', 'қиын', 'дұрыс емес'
        ]

        positive_count = sum(1 for word in positive_keywords if word in feedback_lower)
        negative_count = sum(1 for word in negative_keywords if word in feedback_lower)

        # Sentiment score (-1 to 1)
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        else:
            sentiment_score = 0.0

        # Length factor (longer, more detailed feedback is generally more valuable)
        # Normalize by typical feedback length (assume ~50 chars is average)
        length_factor = min(len(feedback.comment) / 100.0, 1.0)  # Cap at 1.0

        # Specificity factor (presence of specific details, examples, etc.)
        # Multilingual specificity indicators
        specificity_indicators = [
            r'\d+',  # numbers
            # English indicators
            r'example', r'step \d+', r'section \d+', r'because', r'however',
            r'specifically', r'particularly', r'for instance', r'such as',
            # Russian indicators
            r'пример', r'например', r'шаг \d+', r'раздел \d+', r'потому что',
            r'однако', r'конкретно', r'особенно', r'в частности', r'то есть',
            # Kazakh indicators
            r'мысал', r'мысалы', r'қадам \d+', r'бөлім \d+', r'себебі',
            r'алайда', r'нақты', r'әсіресе', r'атап айтқанда'
        ]

        specificity_score = 0
        for pattern in specificity_indicators:
            if re.search(pattern, feedback_lower):
                specificity_score += 0.2  # Each indicator adds 0.2

        specificity_score = min(specificity_score, 1.0)  # Cap at 1.0

        # Combine all components
        sentiment_component = sentiment_score * sentiment_weight
        length_component = length_factor * length_weight
        specificity_component = specificity_score * specificity_weight

        # Calculate final reward
        raw_reward = base_reward + sentiment_component + length_component + specificity_component

        # Normalize to [-1, 1] range if requested
        if normalize:
            # The maximum possible raw reward would be:
            # base(1.0) + sentiment(sentiment_weight) + length(length_weight) + specificity(specificity_weight)
            max_possible = 1.0 + sentiment_weight + length_weight + specificity_weight

            final_reward = np.clip(raw_reward / max_possible, -1.0, 1.0)
        else:
            final_reward = raw_reward

        # Return detailed breakdown
        return final_reward

    def _determine_training_type(self, reward: float) -> str:
        """Determine training strategy based on reward"""
        if reward > 0.5:
            return "positive_reinforcement"  # Good responses - reinforce
        elif reward < -0.5:
            return "negative_correction"     # Bad responses - learn from correction
        else:
            return "neutral_refinement"      # Average responses - slight adjustment