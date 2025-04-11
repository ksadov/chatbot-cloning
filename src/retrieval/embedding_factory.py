from pathlib import Path
from typing import Any, Dict

from src.retrieval.embedding_core import EmbeddingStore
from src.retrieval.local_embedding_store import LocalEmbeddingStore
from src.retrieval.zilliz_embedding_store import ZillizEmbeddingStore


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
                embedding_config_path=Path(config.get("embedding_config_path")),
                vector_dimension=config.get("vector_dimension", 1024),
                document_path=(
                    Path(config.get("document_path"))
                    if config.get("document_path")
                    else None
                ),
                allow_update=config.get("allow_update", True),
                n_results=config.get("n_results", 5),
            )
        elif store_type == "zilliz":
            return ZillizEmbeddingStore(
                embedding_config_path=Path(config.get("embedding_config_path")),
                uri=config.get("uri"),
                token=config.get("token"),
                collection_name=config.get("collection_name"),
                dimension=config.get("vector_dimension"),
                default_n_results=config.get("n_results"),
                document_path=Path(config.get("document_path")),
            )
        else:
            raise ValueError(f"Unsupported embedding store type: {store_type}")
