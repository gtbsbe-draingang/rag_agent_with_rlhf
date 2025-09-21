-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

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