from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class EmbeddingStore(ABC):
    """Abstract base class for embedding stores."""

    @abstractmethod
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
        pass

    @abstractmethod
    def update(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new document to the embedding store.

        Args:
            document: The document text to add
            metadata: Optional metadata for the document

        Returns:
            Boolean indicating if the update was successful
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the embedding store is available and return status information.

        Returns:
            Dictionary containing status information
        """
        pass
