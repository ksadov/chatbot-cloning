import os
from ragatouille import RAGPretrainedModel


def prep_document(document_path, delimiter="\n-----\n"):
    with open(document_path, "r") as file:
        full_document = file.read()
    documents = full_document.split(delimiter)
    return documents


def init_RAGatouille(config):
    print("Loading RAG...")
    index_name = config['index_name']
    rag_fname = f".ragatouille/colbert/indexes/{index_name}"
    if not os.path.exists(rag_fname):
        documents = prep_document(config['document_path'])
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        RAG.index(
            collection=documents,
            index_name=index_name,
            max_document_length=180,
            split_documents=True
        )
    RAG = RAGPretrainedModel.from_index(rag_fname)
    return RAG


class RAGModule:
    def __init__(self, config, index_type):
        self.config = config
        self.index_type = index_type
        if index_type == "RAGatouille":
            self.RAG = init_RAGatouille(config)
        else:
            raise ValueError(f"Index type {index_type} not supported")

    def search(self, query, k=1):
        return self.RAG.search(query=query, k=k)
