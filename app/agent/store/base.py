'Data layer module'

import chainlit.data as cl_data
import chainlit as cl

from typing import Optional, Dict, List
from chainlit.user import PersistedUser, User
from chainlit.element import ElementDict
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    PaginatedResponse,
    Pagination,
    PageInfo,
    ThreadDict,
    ThreadFilter,
)

# from src.libs.data_base.new_dabase import load_threads

from datetime import datetime

class CustomeDataLayer(cl_data.BaseDataLayer):

    async def get_user(self, identifier: str) -> Optional["PersistedUser"]:
        return PersistedUser(
            identifier=identifier,
            display_name=identifier,
            metadata={},
            id=identifier,
            createdAt=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    async def create_user(self, user: "User") -> Optional["PersistedUser"]:
        pass

    async def delete_feedback(self, feedback_id: str) -> bool:
        pass

    async def upsert_feedback(self, feedback: Feedback) -> str:
        pass

    async def create_element(self, element: "Element"):
        pass

    async def get_element(self, thread_id: str, element_id: str) -> Optional["ElementDict"]:
        pass

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass

    async def create_step(self, step_dict: "StepDict"):
        pass

    async def update_step(self, step_dict: "StepDict"):
        pass

    async def delete_step(self, step_id: str):
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        return ""

    async def delete_thread(self, thread_id: str):
        pass

    async def list_threads(self, pagination: "Pagination", filters: "ThreadFilter") -> "PaginatedResponse[ThreadDict]":
        now = datetime.utcnow().isoformat() + "Z"

        dummy_threads: List[ThreadDict] = [
            {
                "id": "thread-001",
                "createdAt": now,
                "name": "First Thread",
                "userId": "user-001",
                "userIdentifier": "andres",
                "tags": ["test", "demo"],
                "metadata": {"project": "dummy"},
                "steps": [
                    {
                        "id": "step-001",
                        "threadId": "thread-001",
                        "name": "Step One",
                        "type": "message",
                        "input": "What's the status?",
                        "output": "All systems operational.",
                        "streaming": False,
                        "metadata": {},
                        "createdAt": now
                    }
                ],
                "elements": [
                    {
                        "id": "element-001",
                        "threadId": "thread-001",
                        "type": "text",
                        "name": "Sample Text Block",
                        "display": "inline",
                        "size": "medium",
                        "language": "en",
                        "page": 1,
                        "props": {"highlight": True},
                        "mime": "text/plain"
                    }
                ]
            }
        ]

        page_info = PageInfo(
            hasNextPage=False,
            startCursor="thread-001",
            endCursor="thread-001"
        )

        return PaginatedResponse(pageInfo=page_info, data=dummy_threads)

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        # session = cl.user_session.get("user").identifier
        # threads: List[ThreadDict] = load_threads(session)

        # for thread in threads:
        #     if thread["id"] == thread_id:
        #         return thread

        now = datetime.utcnow().isoformat() + "Z"

        return {
                "id": "thread-001",
                "createdAt": now,
                "name": "First Thread",
                "userId": "user-001",
                "userIdentifier": "andres",
                "tags": ["test", "demo"],
                "metadata": {"project": "dummy"},
                "steps": [
                    {
                        "id": "step-001",
                        "threadId": "thread-001",
                        "name": "Step One",
                        "type": "message",
                        "input": "What's the status?",
                        "output": "All systems operational.",
                        "streaming": False,
                        "metadata": {},
                        "createdAt": now,
                    }
                ],
                "elements": [
                    {
                        "id": "element-001",
                        "threadId": "thread-001",
                        "type": "text",
                        "name": "Sample Text Block",
                        "display": "inline",
                        "size": "medium",
                        "language": "en",
                        "page": 1,
                        "props": {"highlight": True},
                        "mime": "text/plain"
                    }
                ]
            }

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        pass

    async def build_debug_url(self) -> str:
        pass