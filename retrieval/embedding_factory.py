from typing import Dict, Any
from pathlib import Path

from retrieval.embedding_core import EmbeddingStore
from retrieval.local_embedding_store import LocalEmbeddingStore


class EmbeddingStoreFactory:
    """Factory for creating embedding store instances based on config."""

    @staticmethod
    def create_store(config: Dict[str, Any]) -> EmbeddingStore:
        """
        Create an embedding store instance based on the provided configuration.

        Args:
            config: Configuration dictionary with store parameters

        Returns:
            An EmbeddingStore instance

        Raises:
            ValueError: If the store type is not supported
        """
        store_type = config.get("type", "local")

        if store_type == "local":
            return LocalEmbeddingStore(
                index_path=Path(config.get("index_path", ".vector_store/index")),
                embedding_model_name=config.get(
                    "embedding_model_name", "BAAI/bge-large-en-v1.5"
                ),
                vector_dimension=config.get("vector_dimension", 1024),
                document_path=(
                    Path(config.get("document_path"))
                    if config.get("document_path")
                    else None
                ),
                allow_update=config.get("allow_update", True),
                n_results=config.get("n_results", 5),
            )
        else:
            raise ValueError(f"Unsupported embedding store type: {store_type}")
