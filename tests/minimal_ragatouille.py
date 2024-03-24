from ragatouille import RAGPretrainedModel


def main():
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    documents = [f"document {i}" for i in range(5000)]
    RAG.index(
        collection=documents,
        index_name="demo",
    )

    for i in range(10):
        new_documents = [f"wefwfvaeves {i}"]

        # Add documents to the index
        RAG.add_to_index(new_documents)
        result = RAG.search(query="wefwfvaeves {i}", k=3)
        print("RESULTS", result)

    RAG2 = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/demo/")
    results2 = RAG2.search(query="document 1", k=3)
    print("RESULTS from loaded index", results2)
    results3 = RAG2.search(query="wefwfvaeves 1", k=3)
    print("RESULTS from loaded index", results3)


if __name__ == "__main__":
    main()
