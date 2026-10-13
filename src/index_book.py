import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import NomicEmbeddings

from ingest import (
    BOOK_PATH,
    load_book,
    clean_gutenberg_text,
    extract_sections,
    chunk_sections,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = PROJECT_ROOT / "chroma_db" / "frankenstein"
COLLECTION_NAME = "frankenstein"


def prepare_documents() -> list[Document]:
    """Load, clean, split, and convert the book into LangChain documents."""

    raw_text = load_book(BOOK_PATH)
    cleaned_text = clean_gutenberg_text(raw_text)
    sections = extract_sections(cleaned_text)
    chunks = chunk_sections(sections)

    documents = []

    for chunk in chunks:
        metadata = {
            key: value
            for key, value in chunk.items()
            if key != "text"
        }

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return documents


def build_index() -> Chroma:
    """Create a fresh persistent vector index for Frankenstein."""

    documents = prepare_documents()

    if not documents:
        raise ValueError("No documents were generated.")

    # Rebuild only this book's index when explicitly running this script.
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)

    embeddings = NomicEmbeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    ids = [
        document.metadata["chunk_id"]
        for document in documents
    ]

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    return vector_store


def main() -> None:
    vector_store = build_index()

    print("Vector index created successfully.")
    print(f"Book: {BOOK_PATH.name}")
    print(f"Index location: {CHROMA_PATH}")
    print(f"Indexed chunks: {vector_store._collection.count()}")

    query = "Why did Victor Frankenstein create the creature?"

    print(f"\nTest query: {query}\n")

    results = vector_store.similarity_search(
        query,
        k=3,
    )

    for rank, document in enumerate(results, start=1):
        print(
            f"Result {rank} | "
            f"{document.metadata['section_title']} | "
            f"Chunk {document.metadata['chunk_index']}"
        )
        print(document.page_content)
        print("-" * 70)


if __name__ == "__main__":
    main()
