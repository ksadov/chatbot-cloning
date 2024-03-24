from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    Document,
    Settings,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
import faiss


def test_HuggingFaceEmbedding():
    model_name = "BAAI/bge-large-en-v1.5"
    vector_dimension = 1024
    index_name = "demo"
    rag_fname = f".llama_index/{model_name}/indexes/{index_name}"
    embed_model = HuggingFaceEmbedding(
        model_name=model_name)
    Settings.embed_model = embed_model
    print(
        f"Creating index at {rag_fname}")
    faiss_index = faiss.IndexFlatL2(vector_dimension)
    documents = [f"document {i}" for i in range(5000)]
    documents = [Document(text=doc) for doc in documents]
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    index.storage_context.persist(persist_dir=rag_fname)
    rag_module = index.as_retriever(similarity_top_k=1)

    query1 = "document 1"
    retrieved1 = rag_module.retrieve(query1)[0].text
    print("RESULTS 1", retrieved1)

    for i in range(10):
        index.insert(Document(text=f"teststring {i}"))
        index.storage_context.persist(persist_dir=rag_fname)
        query = f"teststring {i}"
        retrieved = rag_module.retrieve(query)[0].text
        print("new RESULTS", retrieved)
        old_query = f"document {i}"
        retrieved = rag_module.retrieve(old_query)[0].text
        print("old RESULTS", retrieved)

    # load index2 from disk
    print(f"Loading index from {rag_fname}")
    storage_context_2 = StorageContext.from_defaults(
        vector_store=FaissVectorStore.from_persist_dir(rag_fname),
        persist_dir=rag_fname
    )
    index2 = load_index_from_storage(storage_context=storage_context_2)
    rag_module2 = index2.as_retriever(similarity_top_k=1)
    for i in range(10):
        query = f"teststring {i}"
        retrieved = rag_module2.retrieve(query)[0].text
        print("new RESULTS from loaded index", retrieved)
        old_query = f"document {i}"
        retrieved = rag_module2.retrieve(old_query)[0].text
        print("old RESULTS from loaded index", retrieved)


if __name__ == "__main__":
    test_HuggingFaceEmbedding()
