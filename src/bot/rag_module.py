from typing import List

import requests


class RagModule:
    def __init__(self, vector_store_endpoint: str):
        self.vector_store_endpoint = vector_store_endpoint

    def search(self, query: str) -> List[str]:
        response = requests.post(
            f"{self.vector_store_endpoint}/api/search",
            json={"query": query, "n_results": 5},
        )
        response.raise_for_status()
        response_texts = [result["text"] for result in response.json()["results"]]
        return response_texts

    def update(self, query: str) -> None:
        response = requests.post(
            f"{self.vector_store_endpoint}/api/update",
            json={"document": query},
        )
        response.raise_for_status()
