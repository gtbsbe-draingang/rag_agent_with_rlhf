<!--
from nltk.corpus.reader import documentsfrom agent_container.agent.core.rag_agent import RAGAgentfrom agent_container.agent.graph.workflow import create_workflow
-->

# 🧠 RAG Agent using LangGraph & LangChain

This project is a **Retrieval-Augmented Generation (RAG) Agent** with **Reinforcement Learning Loop** built on:

- 🔁 **LangGraph**: for orchestrating recursive retrieval using conditional edges  
- 🦜 **LangChain**: to integrate vector stores, embeddings, memory, and agents  
- 🧭 **Tavily Search**: to augment answers with live web results  
- 🤖 **Ollama Local Models**: as the core LLM (e.g., LLaMA 3, KazLLM)  
- 🔍 **PGVector**: for efficient local semantic retrieval  
- 💡 **HuggingFace Embeddings**: to embed documents and queries for semantic search
- 🦥 **Unsloth and TRL**: for Reinforcement Learning Loop performance boost and fine-tuning

---

## 🚀 Getting Started
### 1. Make sure documents are set up
1. Create folder app/docs
2. Add all documents you want to use

### 2. Start app
Start in containers
   1. Go to _/root/base_ (having _/root/app_)
   2. Open console and write: ```docker build . -t base_docker:latest```
   3. Go to _/root_ (having _/root/app_)
   4. Open console and write: ```docker compose up```

---

## ✨ Features

- **Agent-centric architecture**: all logic encapsulated in a single `Agent` class
Note: for modulation purposes, main 'run' function has be depreciated.
- Basic interface:  
    ```python
    from agent.core.rag_agent import RAGAgent
    from agent.graph import create_workflow

    agent = RAGAgent()
    graph = create_workflow(agent)
    
    # Init vectorstore
    documents_path = "/docs/path"
    agent.init_vectorstores(documents_path)
    
    # All initial state params can be taken from 
    # agent_container/agent/core/state.py
    # Note: question and documents_path are required
    initial_state = {...}
  
    # Run graph
    ans = graph.invoke(initial_state)
    ```
- Recursive document retrieval via **LangGraph** conditional edges
- Seamless integration with **Tavily Search API** for real-time information
- Modular design for easy **extensibility and integration**
- Feedback loop for automatic model adaptation
---

## 🗂️ Project Structure
```graphql
root/
├── base/
│   ├── Dockerfile          # Docker file to initialize base env
│   └── requirements.txt    # Required libraries for base env
├── app/
│   ├── log/
│   │   └──rag_agent.log    # Log file
│   ├── requrements.txt
│   ├── vector_store    # Vector stores folder. Will be created on initialization
│   ├── main.py
│   └── agent/          # Main Agent module: query processing & orchestration
│       ├── docs/      # Source documents to embed
│       ├── utils/
│       │   ├── document_processor.py   # Read/chunk documents tool
│       │   ├── summary_manager.py      # Summary generating tool
│       │   ├── query_analyzer.py       # Query analysis and breakdown tool
│       │   └── logging_config.py       # Logging setup
│       ├── store/
│       │   └── vector_store.py     # FAISS-based vector store logic using LangChain
│       ├── graph/
│       │   └── workflow.py     # LangGraph setup with conditional edges for response
│       │   └── rl_workflow.py     # LangGraph setup with conditional edges for RLHF
│       ├── rl/
│       │   ├── art.py       # ARTe module for trajectories
│       │   ├── entry.py     # Entries for RLHF
│       │   └── training.py  # Main training loop
│       └── core/
│           ├── config.py       # Pydantic config facade
│           ├── state.py        # Graph variables/states
│           ├── agent.py        # Agent class
│           └── settings/
│                   ├── .env.template   # Helps to locate .env file
│                   └── .env            # API keys, settings, and constants
│
├── .dockerignore
├── Dockerfile     # Docker file for main RAG system
├── ollama         # Docker file for ollama system
├── Modelfile      # Modelfile to create a custom model using ollama
├── docker-compose.yml     # Docker-compose for RAG agent container
├── .env  # .env for chainlit key and database
├── .env.template   # Helps to locate .env file
├── requirements.txt    # Main requirements file to setup venv and run code locally
├── .gitignore
├── graphics/           # Image folder
└── README.md
```

---

## 🧠 Workflow

- Uses **LangChain PGVector** to perform local document retrieval
- Embeds documents and queries using **HuggingFace** embedding models
- Implements recursive retrieval with **LangGraph** conditional edge logic
- Falls back to **Tavily Search** when local results are insufficient
- Combines all context and queries **KazLLM/LLaMA 3** for final response and summaries
- Feedback is handled in **LangGraph** conditional edge logic with **CatBoost/Isolation Forest** models for filtration
- When enough feedback collected, starts retraining on **Unsloth**

---

## 🛠️ Customization
All files are modules within my flow, so you there are no direct dependancies.
Follow functions names and return type, you can change every part.
For example:
- VectorStore can be differenct
- Workflow can be redirected or new nodes/edges initialized
- Document preprocessing customized and different extensions added
- Models and other hyperparameters changed

---

## 📌 TODO
- Support multi-turn conversational memory
- Better evaluation technique (fuzzy match)
- Prompt Injection security

---
