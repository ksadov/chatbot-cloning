import os
from pathlib import Path
from tqdm import tqdm

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    Document,
    Settings,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.schema import TextNode
import faiss
import pandas as pd


class RetrievalError(Exception):
    pass


class LocalEmbeddingStore:
    def __init__(
        self,
        index_path: Path,
        embedding_model_name: Path,
        vector_dimension: int,
        document_path: Path,
        allow_update: bool,
        n_results: int,
    ):
        self.index_path = index_path
        self.embedding_model_name = embedding_model_name
        self.vector_dimension = vector_dimension
        self.document_path = document_path
        self.allow_update = allow_update
        self.n_results = n_results
        self.rag_index = init_HuggingFaceEmbedding(
            index_path,
            embedding_model_name,
            vector_dimension,
            document_path,
            allow_update,
        )
        self.rag_module = self.rag_index.as_retriever(similarity_top_k=n_results)

    def search(self, query: str) -> list[str]:
        retrieved = self.rag_module.retrieve(query)
        return [doc.text for doc in retrieved]

    def update(self, document: str):
        if self.allow_update:
            node = Document(text=document)
            self.rag_index.insert(node)
            self.rag_index.storage_context.persist(persist_dir=self.index_path)


def init_HuggingFaceEmbedding(
    index_path: Path,
    embedding_model_name: str,
    vector_dimension: int,
    document_path: Path,
    allow_update: bool,
) -> VectorStoreIndex:
    if allow_update or not os.path.exists(index_path):
        embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = embed_model
    if os.path.exists(index_path):
        print(f"Loading index from {index_path}")
        vector_store = FaissVectorStore.from_persist_dir(index_path)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=index_path
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index
    else:
        print(f"Creating index at {index_path} from {document_path}")
        faiss_index = faiss.IndexFlatL2(vector_dimension)
        if document_path.endswith(".txt"):
            documents = prep_txt_document(document_path)
        else:
            raise ValueError(f"Unsupported document type: {document_path}")
        documents = [Document(text=doc) for doc in documents]
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist(persist_dir=index_path)
    return index


def prep_txt_document(document_path, delimiter="\n-----\n"):
    with open(document_path, "r") as file:
        full_document = file.read()
    documents = full_document.split(delimiter)
    return documents


def test():
    index_path = Path(".vector_store/test_index")
    embedding_model_name = "BAAI/bge-large-en-v1.5"
    vector_dimension = 1024
    document_path = "data/zef.txt"
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
