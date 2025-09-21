"""Feedback memory initialization"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from agent.core.config import RAGConfig


tables = """
-- Basic schema for Chainlit (this may vary by version)
CREATE TABLE IF NOT EXISTS "Thread" (
    id UUID PRIMARY KEY,
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    name TEXT,
    "userId" TEXT,
    metadata JSONB,
    tags TEXT[]
);

CREATE TABLE IF NOT EXISTS "Step" (
    id UUID PRIMARY KEY,
    "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
    "parentId" UUID REFERENCES "Step"(id) ON DELETE CASCADE,
    "startTime" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "endTime" TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    "showInput" TEXT,
    "isError" BOOLEAN,
    type TEXT NOT NULL,
    name TEXT,
    input TEXT,
    output TEXT,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS "Feedback" (
    id UUID PRIMARY KEY,
    "stepId" UUID REFERENCES "Step"(id) ON DELETE CASCADE,
    value INTEGER,
    comment TEXT,
    name TEXT,
    "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

def init_db(config: RAGConfig):
    # Read database URI from environment variable
    database_url = config.pgvector_uri.get_secret_value()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")

    # Initialize SQLAlchemy engine
    engine = create_engine(database_url, echo=True, future=True)

    print("INITIALIZED ENGINE")

    try:
        # Connect and execute
        with engine.begin() as conn:
            conn.execute(text(tables))
        print("Tables initialized successfully.")
    except SQLAlchemyError as e:
        print("Error while initializing database:", e)

if __name__ == "__main__":
    init_db(RAGConfig())
