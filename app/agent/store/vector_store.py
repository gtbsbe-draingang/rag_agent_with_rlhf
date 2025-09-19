"""Vector Store Management utilities."""

import logging
from typing import List, Optional

from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.agent.core.config import RAGConfig


logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(
        self,
        config: RAGConfig,
        collection_name: Optional[str] = None,
    ):
        self.collection_name = collection_name or "default_collection"
        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding_model.get_secret_value()
        )
        self.connection_string = config.pgvector_uri.get_secret_value()
        self.vectorstore: Optional[PGVector] = None

    def exist(self, suffix: str = ''):
        """
        Check if vector store collection exists in PostgreSQL.

        Args:
            suffix (str, optional): suffix to append to the collection name. Defaults to ''.

        Returns:
            bool: True if vector store collection exists, else False.
        """
        try:
            # Try to connect and check if collection exists
            temp_vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embedding,
                collection_name=self.collection_name + suffix,
            )
            # Try to perform a simple operation to check if collection exists
            temp_vectorstore.similarity_search("test", k=1)
            return True
        except Exception:
            return False

    def create(self, documents: List[Document]):
        """Create a new PGVector collection with documents."""
        if not documents:
            logger.warning("No documents provided to create vector store.")
            return

        self.vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embedding,
            connection_string=self.connection_string,
            collection_name=self.collection_name,
        )
        logger.info(f"PGVector collection '{self.collection_name}' created with {len(documents)} documents.")

    def add(self, documents: List[Document]):
        """Add documents to an existing PGVector collection."""
        if not documents:
            logger.warning("No documents provided to add.")
            return

        if self.vectorstore is None:
            if self.exist():
                self.load()
            else:
                self.create(documents)
                return

        self.vectorstore.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to collection '{self.collection_name}'.")

    def delete(self):
        """Delete the PGVector collection."""
        if self.vectorstore is None:
            self.load()

        if self.vectorstore is not None:
            try:
                # PGVector doesn't have a direct delete collection method
                # You might need to implement this based on your PostgreSQL setup
                # For now, we'll clear the vectorstore reference
                self.vectorstore = None
                logger.info(f"Collection '{self.collection_name}' reference cleared.")
                logger.warning("Note: Actual PostgreSQL table deletion may need manual intervention.")
            except Exception as e:
                logger.error(f"Error deleting collection: {e}")
        else:
            logger.info("No collection to delete.")

    def load(self):
        """Load existing PGVector collection."""
        try:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embedding,
                collection_name=self.collection_name,
            )
            logger.info(f"Loaded PGVector collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error loading collection '{self.collection_name}': {e}")
            self.vectorstore = None

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search on the PGVector collection."""
        if self.vectorstore is None:
            logger.error("No PGVector collection loaded.")
            raise ValueError("No PGVector collection loaded.")

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} documents for query.")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

    def retrieve_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        if self.vectorstore is None:
            logger.error("No PGVector collection loaded.")
            raise ValueError("No PGVector collection loaded.")

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Retrieved {len(results)} documents with scores for query.")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search with score: {e}")
            raise e

if __name__ == '__main__':
    config = RAGConfig()
    store = VectorStoreManager(config)
