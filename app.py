# app.py
import os
import anyio
import chainlit as cl
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

load_dotenv()

HISTORY_KEY = "history"
WINDOW = 8
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pass the key explicitly to avoid env/autoload edge cases
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL, temperature=0)

async def assistant_call(history: list[BaseMessage]) -> AIMessage:
    def _run_sync():
        return llm.invoke(history)
    ai_msg = await anyio.to_thread.run_sync(_run_sync)
    return AIMessage(content=getattr(ai_msg, "content", str(ai_msg)))

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(HISTORY_KEY, [])
    await cl.Message("Hi! Chainlit is up. Ask me anything.").send()

@cl.on_message
async def on_message(message: cl.Message):
    history: list[BaseMessage] = cl.user_session.get(HISTORY_KEY, [])
    history.append(HumanMessage(content=message.content))
    history = history[-WINDOW:]

    ai_msg = await assistant_call(history)
    history.append(ai_msg)
    cl.user_session.set(HISTORY_KEY, history)

    await cl.Message(ai_msg.content).send()

# Compatible across versions: no type annotation here
@cl.on_feedback
async def on_feedback(feedback):
    try:
        print("FEEDBACK:", getattr(feedback, "value", None), getattr(feedback, "comment", None))
    except Exception:
        print("FEEDBACK RAW:", feedback)
