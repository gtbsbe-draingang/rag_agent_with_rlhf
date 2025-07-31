"""Main file that opens endpoint for RAG agent querying"""

import logging
import os
import chainlit as cl
from fastapi import FastAPI

from agent.core.rag_agent import RAGAgent
from agent.graph.workflow import create_workflow
from agent.utils.logging_config import setup_logging

# Server Setup
app = FastAPI()
setup_logging()

logger = logging.getLogger(__name__)

# Model Setup
agent = RAGAgent()
graph = create_workflow(agent)
documents_path = "agent/documents"


@cl.password_auth_callback
async def on_authorize(username: str, password: str):
    """Authorization event"""
    if username == os.getenv("ADMIN_ID") and password == os.getenv("ADMIN_PASSWORD"):
        return cl.User(identifier='admin', metadata={'role': 'ADMIN'})


@cl.on_chat_start
async def on_chat_start():
    """Handle chat start"""
    msg = cl.Message(f"Loading database...")
    await msg.send()

    # Initialize vectorstore
    await agent.init_vectorstores(documents_path)

    app_user = cl.user_session.get("user")
    msg.content = f"Hello {app_user.identifier}"
    await msg.update()


@cl.on_message
async def on_msg(msg: cl.Message):
    """Handle messages by sending them into graph"""
    ans = cl.Message("Let's see...")
    await ans.send()

    # Preparation
    initial_state = {
        "question": msg.content,
        "documents_path": documents_path,
        "final_answer": "",
    }
    try:
        result = None
        async for state in graph.astream(initial_state):
            for node_name, node_json in state.items():
                if "additional_info" in node_json:
                    ans.content = node_json["additional_info"]
                    await ans.update()
                if "final_answer" in node_json:
                    result = node_json
        if result:
            agent.store_messages(msg.content, result["final_answer"])

        ans.content = result["final_answer"]
        await ans.update()

    except Exception as e:
        logger.error(e)
        ans.content = "Sorry! There is a problem connecting to our agent. Please try again later."
        await ans.update()
