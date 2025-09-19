"""Configuration management for RAG agent using Pydantic.

Config is loaded and used by pydantic for
data leakage prevention and convenience.
"""

import os
import re

from pydantic import Field, field_validator
from pydantic.types import SecretStr
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings


HOME_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

class RAGConfig(BaseSettings):
    """Configuration for the RAG agent"""

    tavily_api_key: SecretStr = Field(..., description="Tavily API key")

    llm_model: str = Field(default="llama3", description="Ollama model name to use (e.g., 'llama3', 'mistral')")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Base URL for Ollama API")
    llm_temperature: float = Field(default=0.1, ge=0.0, description="Temperature of the LLM model")
    embedding_model: SecretStr = Field(..., description="Embedding model to use")
    embedding_device: str = Field(default="gpu", description="Device to use")
    reranker_model: str = Field(default="BAAI/bge-reranker-large", description="Reranker model to use")

    # Chunking and Retrieval
    chunk_size: int = Field(default=1000, gt=0, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    max_retries: int = Field(default=3, gt=0, description="Maximum number of retries")
    max_retrieval_depth: int = Field(default=2, gt=0, lt=10, description="Maximum retrieval depth")
    top_k_retrieval: int = Field(default=5, gt=0, description="Top K results to retrieve")
    top_n_reranked: int = Field(default=5, gt=0, description="Top N reranked results to retrieve")

    # Compression and Similarity
    compression_enabled: bool = Field(default=True, description="Enable compression")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")

    # Storage
    pgvector_uri: SecretStr = Field(..., description="PGVector connection URI")
    vector_store_path: str = Field(default="./vector_store", description="Path to vector store")

    # Logging
    verbose: bool = Field(default=True, description="Enable verbose logging")

    class Config:
        """Basic configuration for the RAGConfig class"""
        env_file = os.path.join(HOME_DIR, "settings", ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""
        validate_assignment = True

    @field_validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, info: FieldValidationInfo):
        """Ensure chunk_overlap is not larger than chunk_size"""
        if not isinstance(v, int):
            raise TypeError("chunk_overlap must be an integer")
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v


    @field_validator('embedding_model')
    def validate_embeddings_model(cls, v):
        """Ensure embeddings model is not empty"""
        if not v or not v.get_secret_value().strip():
            raise ValueError('Embeddings model cannot be empty')
        return v

    @field_validator('embedding_device')
    def validate_embedding_device(cls, v):
        """Ensure embedding device is correct"""
        if not v in ['cpu', 'cuda']:
            raise ValueError('Embeddings device should be either cpu or cuda (gpu)')
        return v

    @field_validator('vector_store_path')
    def validate_vector_store_path(cls, v):
        """Ensure vector store path is not empty"""
        if not v or not v.strip():
            raise ValueError('Vector store path cannot be empty')
        return v.strip()

    @field_validator('pgvector_uri')
    def validate_pgvector_uri(cls, v):
        """Ensure pgvector uri is not empty and in the right format"""
        if not v or not v.get_secret_value().strip():
            raise ValueError('pgvector uri cannot be empty')

        PG_URI_PATTERN = re.compile(
            r'^postgresql\+psycopg2://'
            r'(?P<user>[^:]+):(?P<password>[^@]+)'  # user:password
            r'@'
            r'(?P<host>[^:/]+)'  # host
            r':'
            r'(?P<port>\d+)'  # port
            r'/'
            r'(?P<dbname>[^/]+)$'  # dbname
        )
        temp = v.get_secret_value().strip()
        if not bool(PG_URI_PATTERN.match(temp)):
            template = "postgresql+psycopg2://<username>:<password>@<host>:<port>/<dbname>"
            raise ValueError(f"PGVector URI is not following required format of '{template}'")

        return v

def get_config() -> RAGConfig:
    """Alternative factory function for backwards compatibility"""
    return RAGConfig()

if __name__ == "__main__":
    # Ensuring working directory
    print(os.getcwd())

    # Functionality test
    config = RAGConfig()
    print(config.model_dump_json(indent=2))