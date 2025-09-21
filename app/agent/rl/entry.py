"""Entries for RLHF"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class FeedbackEntry:
    """Structure for individual feedback entries"""
    question: str
    answer: str
    rating: float  # 1-5 scale
    comment: str
    correct_answer: str = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()