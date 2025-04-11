def prep_txt_document(document_path, delimiter="\n-----\n"):
    """
    Prepare a text document for indexing by splitting it into chunks.

    Args:
        document_path: Path to the text document
        delimiter: Delimiter to split the document by

    Returns:
        List of document chunks
    """
    with open(document_path, "r") as file:
        full_document = file.read()
    documents = full_document.split(delimiter)
    return documents
