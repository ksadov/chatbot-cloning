import requests
import asyncio
from pathlib import Path
import json
from typing import List

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class RemoteEmbeddingModel(BaseEmbedding):
    model_name: str
    api_base: str
    api_key: str
    params: dict

    def __init__(self, model_name: str, api_base: str, api_key: str, params: dict):
        super().__init__(
            model_name=model_name, api_base=api_base, api_key=api_key, params=params
        )
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.params = params

    def _infer(self, text: str) -> List[float]:
        response = requests.post(
            self.api_base,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name, "input": text, **self.params},
        )
        print("Response: ", response.json())
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._infer(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._infer(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(text) for text in texts]


def make_embed_model(config_path: Path) -> BaseEmbedding:
    with open(config_path, "r") as f:
        config = json.load(f)
    if config["api_base"]:
        return RemoteEmbeddingModel(
            config["model_name"],
            config["api_base"],
            config["api_key"],
            config["params"],
        )
    else:
        return HuggingFaceEmbedding(model_name=config["model_name"])
