import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    Document,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
import faiss

from retrieval.embedding_core import EmbeddingStore, RetrievalError
from retrieval.embed_model import make_embed_model


class LocalEmbeddingStore(EmbeddingStore):
    """Implementation of EmbeddingStore for local FAISS indexing and retrieval."""

    def __init__(
        self,
        index_path: Path,
        embedding_config_path: Path,
        vector_dimension: int,
        document_path: Optional[Path] = None,
        allow_update: bool = True,
        n_results: int = 5,
    ):
        """
        Initialize a local embedding store using FAISS.

        Args:
            index_path: Path to store the FAISS index
            embedding_config_path: Path to the embedding config
            vector_dimension: Dimension of the embedding vectors
            document_path: Path to the document to index (if creating a new index)
            allow_update: Whether to allow adding new documents
            n_results: Default number of results to return from search
        """
        self.index_path = index_path
        self.vector_dimension = vector_dimension
        self.document_path = document_path
        self.allow_update = allow_update
        self.default_n_results = n_results
        self.embed_model = make_embed_model(embedding_config_path)
        Settings.embed_model = self.embed_model
        self.rag_index = self._init_embedding_index()
        self.rag_module = self.rag_index.as_retriever(similarity_top_k=n_results)

    def _init_embedding_index(self) -> VectorStoreIndex:
        """Initialize or load the FAISS index."""

        if os.path.exists(self.index_path):
            print(f"Loading index from {self.index_path}")
            vector_store = FaissVectorStore.from_persist_dir(self.index_path)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=self.index_path
            )
            index = load_index_from_storage(storage_context=storage_context)
            return index
        else:
            if not self.document_path:
                raise ValueError(
                    "Document path must be provided when creating a new index"
                )

            print(f"Creating index at {self.index_path} from {self.document_path}")
            faiss_index = faiss.IndexFlatL2(self.vector_dimension)

            if str(self.document_path).endswith(".txt"):
                documents = self._prep_txt_document(self.document_path)
            else:
                raise ValueError(f"Unsupported document type: {self.document_path}")

            documents = [Document(text=doc) for doc in documents]
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            index.storage_context.persist(persist_dir=self.index_path)

        return index

    def _prep_txt_document(self, document_path, delimiter="\n-----\n"):
        """
        Prepare a text document for indexing by splitting it into chunks.

        Args:
            document_path: Path to the text document
            delimiter: Delimiter to split the document by

        Returns:
            List of document chunks
        """
        with open(document_path, "r") as file:
            full_document = file.read()
        documents = full_document.split(delimiter)
        return documents

    def search(
        self, query: str, n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query
            n_results: Number of results to return (overrides default)

        Returns:
            List of dictionaries containing search results with text and metadata
        """
        n = n_results if n_results is not None else self.default_n_results

        # Update retriever if n_results changed
        if n != self.rag_module._similarity_top_k:
            self.rag_module = self.rag_index.as_retriever(similarity_top_k=n)

        try:
            retrieved = self.rag_module.retrieve(query)

            # Convert to a more API-friendly format
            results = []
            for i, node in enumerate(retrieved):
                results.append(
                    {
                        "id": i,
                        "text": node.text,
                        "score": node.score if hasattr(node, "score") else None,
                        "metadata": node.metadata if hasattr(node, "metadata") else {},
                    }
                )

            return results
        except Exception as e:
            raise RetrievalError(f"Error during retrieval: {str(e)}")

    def update(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new document to the embedding store.

        Args:
            document: The document text to add
            metadata: Optional metadata for the document

        Returns:
            Boolean indicating if the update was successful
        """
        if not self.allow_update:
            return False

        metadata = metadata or {}
        node = Document(text=document, metadata=metadata)

        try:
            self.rag_index.insert(node)
            self.rag_index.storage_context.persist(persist_dir=self.index_path)
            return True
        except Exception as e:
            print(f"Error updating index: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the embedding store is available and return status information.

        Returns:
            Dictionary containing status information
        """
        try:
            # Basic check - try to access the index
            _ = self.rag_index.storage_context.vector_store

            return {
                "status": "ok",
                "type": "local",
                "index_path": str(self.index_path),
                "embedding_model": self.embedding_model_name,
                "allow_update": self.allow_update,
                "exists": os.path.exists(self.index_path),
            }
        except Exception as e:
            return {"status": "error", "type": "local", "error": str(e)}


# For testing
def test():
    index_path = Path(".vector_store/test_index")
    embedding_model_name = "BAAI/bge-large-en-v1.5"
    vector_dimension = 1024
    document_path = Path("data/zef.txt")
    allow_update = True
    n_results = 5

    local_embedding_store = LocalEmbeddingStore(
        index_path,
        embedding_model_name,
        vector_dimension,
        document_path,
        allow_update,
        n_results,
    )
    print(local_embedding_store.search("favorite animal"))


if __name__ == "__main__":
    test()
