from sentence_transformers import CrossEncoder 
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from embeddings import NomicEmbeddings

from index_book import (
    CHROMA_PATH,
    COLLECTION_NAME,
    prepare_documents,
)


EMBEDDING_MODEL = "nomic-embed-text"
RETRIEVAL_K = 20
FINAL_K = 5
RERANK_CANDIDATES = 30
RRF_K = 60

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def rerank_results(
    query: str,
    fused_results: list[tuple],
    reranker: CrossEncoder,
    candidate_k: int = RERANK_CANDIDATES,
    final_k: int = FINAL_K,
) -> list[tuple]:
    candidates = fused_results[:candidate_k]

    if not candidates:
        return []

    pairs = [
        (query, document.page_content)
        for document, _rrf_score in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    for (document, rrf_score), score in zip(candidates, scores):
        if document.metadata["chunk_id"] == "frankenstein_07_016":
            print("\n--- Target Reranker Diagnostic ---")
            print(f"RRF score: {rrf_score:.5f}")
            print(f"Reranker score: {float(score):.4f}")
            print(f"Text: {document.page_content}")

    reranked_results = [
        (document, float(score))
        for (document, _rrf_score), score in zip(candidates, scores)
    ]

    reranked_results.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return reranked_results[:final_k]

def load_vector_store() -> Chroma:
    """Load the existing persistent ChromaDB index."""

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            f"Vector index not found at {CHROMA_PATH}. "
            "Run index_book.py first."
        )

    embeddings = NomicEmbeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )



def reciprocal_rank_fusion(
    ranked_lists: list[list],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> list[tuple]:
    """Combine ranked document lists using weighted Reciprocal Rank Fusion."""

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("Each ranked list must have a corresponding weight.")

    if any(weight < 0 for weight in weights):
        raise ValueError("Weights cannot be negative.")

    scores = defaultdict(float)
    documents_by_id = {}

    for ranked_documents, weight in zip(ranked_lists, weights):
        seen_in_list = set()

        for rank, document in enumerate(ranked_documents, start=1):
            document_id = document.metadata["chunk_id"]

            # Prevent duplicate contributions within one ranking.
            if document_id in seen_in_list:
                continue

            seen_in_list.add(document_id)
            documents_by_id[document_id] = document
            scores[document_id] += weight / (k + rank)

    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (documents_by_id[document_id], score)
        for document_id, score in ranked_results
    ]



def hybrid_search(
    query: str,
    vector_store: Chroma,
    bm25_retriever: BM25Retriever,
    k: int = RETRIEVAL_K,
    final_k: int = FINAL_K,
) -> list[tuple]:
    """Retrieve candidates through dense and keyword search, then fuse."""

    dense_results = vector_store.similarity_search(
        query,
        k=k,
    )

    keyword_results = bm25_retriever.invoke(query)[:k]

    fused_results = reciprocal_rank_fusion(
        [dense_results, keyword_results],
        weights=[0.7, 0.3]
    )

    return fused_results[:final_k]


def main() -> None:
    print("Loading book chunks...")

    chunks = prepare_documents()

    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=RETRIEVAL_K,
    )

    vector_store = load_vector_store()

    print(f"Book chunks loaded: {len(chunks)}")
    print(
        f"Vector index contains: "
        f"{vector_store._collection.count()} chunks"
    )
    print(f"Candidates per retriever: {RETRIEVAL_K}")
    print(f"Final hybrid results: {FINAL_K}")

    query = input("\nAsk a question about Frankenstein: ").strip()

    if not query:
        print("Please enter a question.")
        return

    dense_results = vector_store.similarity_search(
        query,
        k=RETRIEVAL_K,
    )

        # Step 1: Retrieve candidates using both methods
    dense_results = vector_store.similarity_search(
        query,
        k=RETRIEVAL_K,
    )

    bm25_results = bm25_retriever.invoke(query)[:RETRIEVAL_K]

    # Step 2: Combine the rankings using RRF
    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )

    # Step 3: Rerank the best RRF candidates
    print("\nLoading cross-encoder reranker...")
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
    )

    target_chunk_id = "frankenstein_07_016"

    print("\n--- Target Passage Diagnostic ---")

    target_rank = None

    for rank, (document, rrf_score) in enumerate(
        fused_results,
        start=1,
    ):
        if document.metadata["chunk_id"] == target_chunk_id:
            target_rank = rank
            print(f"Target found at RRF rank: {rank}")
            print(f"RRF score: {rrf_score:.5f}")
            print(f"Section: {document.metadata['section_title']}")
            print(f"Text: {document.page_content}")
            break

    if target_rank is None:
        print("Target passage is absent from fused candidates.")

    results = rerank_results(
        query=query,
        fused_results=fused_results,
        reranker=reranker,
    )

    # Step 4: Display the final results
    print(f"\nRRF candidates available: {len(fused_results)}")
    print(f"Candidates reranked: {min(len(fused_results), RERANK_CANDIDATES)}")
    print(f"Final results: {len(results)}")

    print("\n--- Reranked Results ---")

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata

        print(f"\nRank: {rank}")
        print(f"Reranker score: {score:.4f}")
        print(f"Section: {metadata['section_title']}")
        print(f"Chunk ID: {metadata['chunk_id']}")
        print(f"Chunk index: {metadata['chunk_index']}")
        print(f"Excerpt: {document.page_content[:600]}")
        print("-" * 80)

    bm25_results = bm25_retriever.invoke(query)[:RETRIEVAL_K]

    print("\n========== DENSE SEARCH ==========\n")

    for rank, doc in enumerate(dense_results, start=1):
        print(
            f"Rank {rank} | "
            f"{doc.metadata['section_title']} | "
            f"Chunk {doc.metadata['chunk_index']}"
        )
        print(doc.page_content[:400])
        print("-" * 60)

    print("\n========== BM25 SEARCH ==========\n")

    for rank, doc in enumerate(bm25_results, start=1):
        print(
            f"Rank {rank} | "
            f"{doc.metadata['section_title']} | "
            f"Chunk {doc.metadata['chunk_index']}"
        )
        print(doc.page_content[:400])
        print("-" * 60)

    results = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )[:FINAL_K]

    print("\n========== HYBRID SEARCH ==========\n")

    for rank, (doc, score) in enumerate(results, start=1):
        print(
            f"Rank {rank} | "
            f"RRF score: {score:.5f} | "
            f"{doc.metadata['section_title']} | "
            f"Chunk {doc.metadata['chunk_index']}"
        )
        print(doc.page_content[:600])
        print("-" * 60)


if __name__ == "__main__":
    main()
