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

## ✨ Features

- **Agent-centric architecture**: all logic encapsulated in a single `Agent` class
Note: for modulation purposes, main 'run' function has be depreciated.
- Basic interface:  
    ```python
    from agent import RAGAgent
    from agent import create_workflow

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
├── app/
│   ├── log/
│   │   └──rag_agent.log    # Log file
│   ├── .dockerignore
│   ├── docker-compose.yaml     # Docker-compose for RAG agent container
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
├── .env  # .env for chainlit key and database
├── .env.template   # Helps to locate .env file
├── requirements.txt    # Main requirements file to setup venv and run code locally
├── .gitignore
├── graphics/           # Image folder
└── README.md
```

---

## 🚀 Getting Started
### 1. Install Requirements
```bash
pip install -r requirements.txt
```
Make sure the following packages are included:
- ```langchain```
- ```langgraph```
- ```pyscopg2```
- ```huggingface_hub```
- ```ollama```
- ```unsloth```
- ```trl```
- ```transformers```

Will be in the future version, now we use LangChain module TavilySearch
- ```tavily-python```

### 2. Set Environment Variables
1. Paste your variables into .env that can be created in **the same folder and the same format** as _.env.template_

### 3. Set up Chainlit
1. Go to /app and open the console
2. Write ```chainlit create-secret```
3. Copy ```CHAINLIT_AUTH_SECRET=...``` into your upper .env file
4. Write ```DATABASE_URL=postgresql://<username>:<password>@<host>:<port>/<dbname>``` into your upper .env file
   
### 4. Make sure documents are set up
1. Create folder /docs
2. Add all documents you want to use

### 5. Start app
1. If you want to start in containers
   1. Go to _/app_
   2. Open console and write: ```docker compose up```
2. If you want to start it locally:
   1. Go to _/root_ folder
   2. Open console and write: ```chainlit run app/app.py```
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
