import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from tqdm import tqdm

from src.retrieval.documents import prep_parquet, prep_txt_document
from src.retrieval.embed_model import make_embed_model
from src.retrieval.embedding_core import EmbeddingStore


class ZillizEmbeddingStore(EmbeddingStore):
    """Implementation of EmbeddingStore for Zilliz Cloud."""

    def __init__(
        self,
        embedding_config_path: Path,
        uri: str,
        token: str,
        collection_name: str,
        dimension: int,
        document_path: Optional[Path],
        default_n_results: int = 5,
    ):
        """
        Initialize Zilliz Cloud connection.

        Args:
            uri: Zilliz Cloud endpoint URI
            token: API token for authentication
            collection_name: Name of the collection to use
            dimension: Dimension of the embedding vectors
            default_n_results: Default number of results to return from search
        """
        self.collection_name = collection_name
        self.document_path = document_path
        self.dimension = dimension
        self.default_n_results = default_n_results
        self.embed_model = make_embed_model(embedding_config_path)

        # Connect to Zilliz Cloud
        print(f"Connecting to Zilliz Cloud at {uri}")
        connections.connect(alias="default", uri=uri, token=token, secure=True)
        print("Connected to Zilliz Cloud")

        # Create collection if it doesn't exist
        if not utility.has_collection(collection_name):
            self._create_collection()
        else:
            self.collection = Collection(collection_name)
        self.collection.load()

    def _create_collection(self):
        """Create a new collection with the specified schema."""
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True
            ),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(
            fields=fields, description="Document embeddings collection"
        )

        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create IVF_FLAT index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Created index for collection {self.collection_name}")

        print(f"document_path: {self.document_path}")
        if self.document_path is not None:
            # check if text file
            if self.document_path.suffix == ".txt":
                documents = prep_txt_document(self.document_path)
            else:
                documents = prep_parquet(self.document_path)
            print(f"Adding {len(documents)} documents to collection")
            for document in tqdm(documents):
                self.update(document)
            print(f"Added {len(documents)} documents to collection")

    def search(
        self, query: str, n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query
            n_results: Number of results to return (overrides default)

        Returns:
            List of dictionaries containing search results
        """
        n = n_results if n_results is not None else self.default_n_results

        # Generate embedding for query
        query_embedding = self.embed_model.get_text_embedding(query)

        # Search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=n,
            output_fields=["text"],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "id": hit.id,
                        "text": hit.entity.get("text"),
                        "score": float(hit.score),
                        "metadata": {},  # Add additional metadata if needed
                    }
                )

        return formatted_results

    def update(self, document: str, metadata: Dict[str, Any] = {}) -> bool:
        """
        Add a new document to the embedding store.

        Args:
            document: The document text to add
            metadata: Optional metadata for the document

        Returns:
            Boolean indicating if the update was successful
        """
        try:
            # Generate embedding
            embedding = self.embed_model.get_text_embedding(document.text)

            # Prepare data
            data = {
                "id": str(uuid.uuid4()),
                "embedding": embedding,
                "text": document.text,
                "metadata": document.metadata,
            }

            # Insert into collection
            self.collection.insert(data)
            return True

        except Exception as e:
            print(f"Error updating document: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the embedding store is available and return status information.

        Returns:
            Dictionary containing status information
        """
        try:
            stats = self.collection.stats()
            return {
                "status": "ok",
                "type": "zilliz",
                "collection_name": self.collection_name,
                "row_count": stats["row_count"],
                "index_type": "IVF_FLAT",
            }
        except Exception as e:
            return {"status": "error", "type": "zilliz", "error": str(e)}
