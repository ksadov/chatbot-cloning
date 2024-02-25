import os
from ragatouille import RAGPretrainedModel


def prep_document(document_path, delimiter="\n-----\n"):
    with open(document_path, "r") as file:
        full_document = file.read()
    documents = full_document.split(delimiter)
    return documents


def setup_RAG(index_name, documents):
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG.index(
        collection=documents,
        index_name=index_name,
        max_document_length=180,
        split_documents=True
    )


def init_RAG(config):
    print("Loading RAG...")
    index_name = config['index_name']
    rag_fname = f".ragatouille/colbert/indexes/{index_name}"
    if not os.path.exists(rag_fname):
        documents = prep_document(config['document_path'])
        setup_RAG(index_name, documents)
    RAG = RAGPretrainedModel.from_index(rag_fname)
    return RAG
