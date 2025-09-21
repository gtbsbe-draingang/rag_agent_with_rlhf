"""Main file that opens endpoint for RAG agent querying"""

import logging
import os
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from mcp.server.fastmcp.prompts.base import UserMessage

from app.agent.rl.entry import FeedbackEntry
from initialize_memory import init_db
from agent.store.base import CustomeDataLayer
from agent.core.rag_agent import RAGAgent
from agent.graph import create_workflow, create_rl_workflow
from agent.utils.logging_config import setup_logging

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

# Model Setup
agent = RAGAgent()
graph = create_workflow(agent)
rl_graph = create_rl_workflow(agent)
documents_path = "app/agent/docs"

# Simple History
HISTORY_KEY = "history"
WINDOW = 8


@cl.password_auth_callback
async def on_authorize(username: str, password: str):
    """Authorization event"""
    if username == "admin" and password == "admin":
        return cl.User(identifier='admin', metadata={'role': 'ADMIN'})


@cl.on_chat_start
async def on_chat_start():
    """Handle chat start"""
    msg = cl.Message(f"Initializing context tables...")
    await msg.send()

    # Initialize feedback tables
    init_db(config=agent.config)

    msg.content = f"Initializing knowledge base..."
    await msg.update()

    # Initialize vectorstore
    await agent.init_vectorstores(documents_path)

    app_user = cl.user_session.get("user")
    msg.content = f"Hello {app_user.identifier}"
    await msg.update()


@cl.on_message
async def on_message(msg: cl.Message):
    history: list[BaseMessage] = cl.user_session.get(HISTORY_KEY, [])
    history.append(HumanMessage(content=msg.content))
    history = history[-WINDOW:]

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

    history.append(AIMessage(content=ans.content))
    cl.user_session.set(HISTORY_KEY, history)

# (RLHF) REINFORCEMENT LEARNING HUMAN FEEDBACK LOOP
@cl.on_feedback
async def on_feedback(feedback):
    answer = ""
    history: list[BaseMessage] = cl.user_session.get(HISTORY_KEY, [])
    for msg in history[::-1]:
        if isinstance(msg, AIMessage):
            answer = msg.content
            break

    value, comment = None, None
    try:
        value, comment = getattr(feedback, "value", None), getattr(feedback, "comment", None)
    except Exception as e:
        logger.error(f"Incorrect feedback format: {e}")

    if value is not None:
        try:
            initial_state = FeedbackEntry(
                answer=answer,
                rating=int(value),
                comment=comment
            )

            result = None
            async for state in rl_graph.astream(initial_state):
                for node_name, node_json in state.items():
                    if "result" in node_json:
                        result = node_json["result"]
            if result:
                logger.info("Model has been retrained!")
        except Exception as e:
            logger.error(f"Problem with RLHF loop: {e}")

# DATA LAYER
@cl.data_layer
def get_data_layer():
    """
    Returns an instance of CustomeDataLayer.
    """
    return CustomeDataLayer()
