import os

from ragatouille import RAGPretrainedModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    Document,
    Settings,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
import datetime
import faiss
import pandas as pd


def prep_txt_document(document_path, delimiter="\n-----\n"):
    with open(document_path, "r") as file:
        full_document = file.read()
    documents = full_document.split(delimiter)
    return documents


def make_metadata_dict(entry):
    metadata_fields = ["conversation", "author", "timestamp"]
    metadata = {field: entry[field] for field in metadata_fields}
    return metadata


def prep_parquet_documents(document_path, alias_dict, include_timestamp):
    df = pd.read_parquet(document_path)
    # convert to list of dictionaries
    entries = df.to_dict(orient="records")
    documents = [entry["content"] for entry in entries]
    metadatas = [make_metadata_dict(entry) for entry in entries]
    # check if alias_field is in metadata
    for i, metadata in enumerate(metadatas):
        if metadata['author'] in alias_dict:
            metadata['author'] = alias_dict[str(metadata['author'])]
        if include_timestamp:
            timestamp_prefix = f"[{metadata['timestamp'].strftime('%Y-%m-%d %H:%M')}]"
        else:
            timestamp_prefix = ""
        documents[i] = f"{timestamp_prefix} {metadata['author']}: {documents[i]}"
    return documents, metadatas


def init_RAGatouille(index_info, include_timestamp):
    print("Initializing RAG...")
    index_name = index_info['index_name']
    rag_fname = f".ragatouille/colbert/indexes/{index_name}"
    if not os.path.exists(rag_fname):
        print(
            f"Creating index at {rag_fname} from {index_info['document_path']}")
        if index_info['document_path'].endswith(".txt"):
            documents = prep_txt_document(index_info['document_path'])
        elif index_info['document_path'].endswith(".parquet"):
            documents = prep_parquet_documents(
                index_info['document_path'], index_info['alias_dict'], include_timestamp)
        else:
            raise ValueError("Document path must end with .txt or .parquet")
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        RAG.index(
            collection=documents,
            index_name=index_name,
            max_document_length=180,
            split_documents=True
        )
    RAG = RAGPretrainedModel.from_index(rag_fname)
    return RAG


def init_HuggingFaceEmbedding(index_info, k, include_timestamp):
    print("Initializing HuggingFaceEmbedding...")
    model_name = index_info['model_name']
    vector_dimension = index_info['vector_dimension']
    index_name = index_info['index_name']
    rag_fname = f".llama_index/{model_name}/indexes/{index_name}"
    embed_model = HuggingFaceEmbedding(
        model_name=model_name)
    Settings.embed_model = embed_model
    if not os.path.exists(rag_fname):
        print(
            f"Creating index at {rag_fname} from {index_info['document_path']}")
        faiss_index = faiss.IndexFlatL2(vector_dimension)
        if index_info['document_path'].endswith(".txt"):
            documents = prep_txt_document(index_info['document_path'])
        else:
            documents, _ = prep_parquet_documents(
                index_info['document_path'], index_info['alias_dict'], include_timestamp=include_timestamp)
        documents = [Document(text=doc) for doc in documents]
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        index.storage_context.persist(persist_dir=rag_fname)
    else:
        vector_store = FaissVectorStore.from_persist_dir(rag_fname)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=rag_fname
        )
        index = load_index_from_storage(storage_context=storage_context)
    return index.as_retriever(similarity_top_k=k)


class RAGModule:
    def __init__(self, config, k):
        self.config = config
        self.k = k
        self.index_info = config['index_info']
        self.include_timestamp = config['include_timestamp']
        if self.index_info['type'] == "RAGatouille":
            self.rag_module = init_RAGatouille(
                self.index_info, self.include_timestamp)
        elif self.index_info['type'] == "HuggingFaceEmbedding":
            self.rag_module = init_HuggingFaceEmbedding(
                self.index_info, k, self.include_timestamp)

    def search(self, query):
        if self.index_info['type'] == "RAGatouille":
            retrieved = self.rag_module.search(query=query, k=self.k)
            return [doc['content'] for doc in retrieved]
        else:
            retrieved = self.rag_module.retrieve(query)
            return [doc.text for doc in retrieved]
