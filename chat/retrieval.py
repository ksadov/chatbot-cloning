import os
from tqdm import tqdm

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
from llama_index.core.schema import TextNode
import faiss
import pandas as pd


class RetrievalError(Exception):
    pass


def prep_txt_document(document_path, delimiter="\n-----\n"):
    with open(document_path, "r") as file:
        full_document = file.read()
    documents = full_document.split(delimiter)
    return documents


def make_metadata_dict(entry):
    metadata_fields = ["conversation", "user_id", "timestamp"]
    metadata = {field: entry[field] for field in metadata_fields}
    return metadata


def load_parquet_documents(document_path, alias_dict, include_timestamp, name):
    df = pd.read_parquet(document_path)
    # convert to list of dictionaries
    entries = df.to_dict(orient="records")
    documents = [entry["content"].strip() for entry in entries]
    metadatas = [make_metadata_dict(entry) for entry in entries]
    # check if alias_field is in metadata
    for i, metadata in enumerate(metadatas):
        if metadata['user_id'] is None:
            metadata['user_id'] = name
        if metadata['user_id'] in alias_dict:
            metadata['user_id'] = alias_dict[str(metadata['user_id'])]
        if include_timestamp:
            timestamp = pd.to_datetime(
                metadata['timestamp']).strftime('%Y-%m-%d %H:%M')
            # check if timestamp is before 2000
            if timestamp.split("-")[0] < "2000":
                timestamp = f"Unknown Timestamp"
            timestamp_prefix = f"[{timestamp}]"
        else:
            timestamp_prefix = ""
        documents[i] = f"{timestamp_prefix} {metadata['user_id']}: {documents[i]}"
    return documents, metadatas


def prep_parquet_documents(document_path, alias_dict, include_timestamp, name):
    # check if there are seperate /chat and /blog subdirectories
    if os.path.exists(os.path.join(document_path, "blog")):
        chat_documents, chat_metadatas = load_parquet_documents(
            os.path.join(
                document_path, "chat"), alias_dict, include_timestamp, name
        )
    if os.path.exists(os.path.join(document_path, "blog")):
        blog_documents, blog_metadatas = load_parquet_documents(
            os.path.join(
                document_path, "blog"), alias_dict, include_timestamp, name
        )
    else:
        # assuming all are chat documents
        chat_documents, chat_metadatas = load_parquet_documents(
            document_path, alias_dict, include_timestamp)
        blog_documents, blog_metadatas = [], []
    return chat_documents, chat_metadatas, blog_documents, blog_metadatas


def filter_and_chunk(documents, metadatas, chunk_depth, name=None, overlap=0):
    print("ORIGINAL LENGTH: ", len(documents))
    filtered_documents = []
    filtered_metadatas = []
    # partition documents by conversation
    unique_conversations = list(set([metadata["conversation"]
                                     for metadata in metadatas]))
    for conversation in tqdm(unique_conversations):
        conv_ids = [i for i, metadata in enumerate(
            metadatas) if metadata["conversation"] == conversation]
        # get indexes of documents that contain alias
        if name is not None:
            chunking_indexes = [i for i in conv_ids if name in documents[i]]
        else:
            # indexes should be spaced by overlap
            chunking_indexes = conv_ids[::chunk_depth - overlap]
        # for each document at a chunking index, create a filtered document containting of the previous chunk_depth documents
        for i in chunking_indexes:
            if i - chunk_depth < 0:
                start = 0
            else:
                start = i + 1 - chunk_depth
            chunk_doc = []
            chunk_metadatas = []
            for j in range(start, i + 1):
                chunk_doc.append(documents[j])
                chunk_metadatas.append(metadatas[j])
            filtered_documents.append(("\n").join(chunk_doc))
            filtered_metadatas.append(chunk_metadatas)
    print("FILTERED AND CHUNKED LENGTH: ", len(filtered_documents))
    return filtered_documents, filtered_metadatas


def format_blog(blog_documents, blog_metadatas, name, include_timestamp):
    formatted_blog = []
    for doc, meta in tqdm(zip(blog_documents, blog_metadatas), total=len(blog_documents)):
        meta['user_id'] = name
        if include_timestamp:
            if meta['timestamp'] is not None:
                timestamp = pd.to_datetime(
                    meta['timestamp']).strftime('%Y-%m-%d %H:%M')
                # check if timestamp is before 2000
                if timestamp.split("-")[0] < "2000":
                    timestamp = f"Unknown Timestamp"
            else:
                timestamp = f"Unknown Timestamp"
            timestamp_prefix = f"[{timestamp}]"
        else:
            timestamp_prefix = ""
        formatted_blog.append(f"{timestamp_prefix} {name}: {doc}")
    return formatted_blog, blog_metadatas


def init_RAGatouille(index_info, name, include_timestamp):
    print("Initializing RAG...")
    index_name = index_info['index_name']
    rag_fname = f".ragatouille/colbert/indexes/{index_name}"
    if not os.path.exists(rag_fname):
        print(
            f"Creating index at {rag_fname} from {index_info['document_path']}")
        if index_info['document_path'].endswith(".txt"):
            documents = prep_txt_document(index_info['document_path'])
        elif index_info['document_path'].endswith(".parquet"):
            chat_documents, chat_metadata, blog_documents, blog_metadata = prep_parquet_documents(
                index_info['document_path'], index_info['alias_dict'], include_timestamp, name
            )
            if index_info['overlap'] is None:
                name = name
            else:
                name = None
            chat_documents, chat_metadata = filter_and_chunk(
                chat_documents, chat_metadata, index_info['chunk_depth'], name=name, overlap=index_info['overlap'])
            documents = chat_documents + blog_documents
            metadata = chat_metadata + blog_metadata
        else:
            raise ValueError("Document path must end with .txt or .parquet")
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        RAG.index(
            collection=documents,
            metadata=metadata,
            index_name=index_name,
            max_document_length=180,
            split_documents=True
        )
    RAG = RAGPretrainedModel.from_index(rag_fname)
    return RAG, rag_fname


def init_HuggingFaceEmbedding(index_info, name, include_timestamp):
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
            documents, metadata, blog_documents, blog_metadata = prep_parquet_documents(
                index_info['document_path'], index_info['alias_dict'], include_timestamp=include_timestamp, name=name
            )
            if index_info['overlap'] is None:
                name = name
            else:
                name = None
            chat_documents, chat_metadata = filter_and_chunk(
                documents, metadata, index_info['chunk_depth'], name=name, overlap=index_info['overlap']
            )
            documents = chat_documents + blog_documents
            metadata = chat_metadata + blog_metadata
        documents = [Document(text=doc) for doc in documents]
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        print("Creating index...")
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
    return index, rag_fname


class RAGModule:
    def __init__(self, config, k):
        self.config = config
        self.k = k
        self.index_info = config['index_info']
        self.include_timestamp = config['include_timestamp']
        if self.index_info['type'] == "RAGatouille":
            self.rag_module, self.rag_fname = init_RAGatouille(
                self.index_info, config['name'], self.include_timestamp)
        elif self.index_info['type'] == "HuggingFaceEmbedding":
            self.rag_index, self.rag_fname = init_HuggingFaceEmbedding(
                self.index_info, config['name'], self.include_timestamp)
            self.rag_module = self.rag_index.as_retriever(similarity_top_k=k)

    def search(self, query):
        if self.index_info['type'] == "RAGatouille":
            retrieved = self.rag_module.search(query=query, k=self.k)
            return [doc['content'] for doc in retrieved]
        else:
            retrieved = self.rag_module.retrieve(query)
            return [doc.text for doc in retrieved]

    def update(self, document):
        if self.index_info['type'] == "RAGatouille":
            raise NotImplementedError("RAGatouille does not support updating")
        else:
            node = Document(text=document)
            self.rag_index.insert(node)
            self.rag_index.storage_context.persist(persist_dir=self.rag_fname)
