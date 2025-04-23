from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel


class EmbedDocument(BaseModel):
    text: str
    metadata: Dict[str, str]


def prep_txt_document(
    document_path: Path, delimiter: str = "\n-----\n"
) -> List[EmbedDocument]:
    """
    Prepare a text document for indexing by splitting it into chunks.

    Args:
        document_path: Path to the text document
        delimiter: Delimiter to split the document by

    Returns:
        List of dictionaries containing the document chunks, where each dictionary has the following keys:
        - "text": The text of the document chunk
        - "metadata" (optional): The metadata of the document chunk, in JSON format
    """
    with open(document_path, "r") as file:
        full_document = file.read()
    document_texts = full_document.split(delimiter)
    documents = [EmbedDocument(text=text, metadata={}) for text in document_texts]
    return documents


def prep_parquet(document_path: Path) -> List[Dict]:
    """
    Prepare a parquet document or directory of parquet documents for indexing.

    Args:
        document_path: Path to the parquet document or directory of parquet documents

    Returns:
        List of dictionaries containing the document chunks, where each dictionary has the following keys:
        - "text": The text of the document chunk
        - "metadata" (optional): The metadata of the document chunk, in JSON format
    """

    documents = []

    # Handle both single file and directory cases
    parquet_files = (
        [document_path]
        if document_path.is_file()
        else list(document_path.glob("*.parquet"))
    )

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        # Convert each row to a dictionary
        for _, row in df.iterrows():
            doc_text = row.get("text", "")
            doc_meta = row.get("metadata", {})
            # stringify metadata
            doc_meta = {k: str(v) for k, v in doc_meta.items()}
            documents.append(EmbedDocument(text=doc_text, metadata=doc_meta))

    return documents
