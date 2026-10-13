
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class NomicEmbeddings(Embeddings):
    """Apply Nomic prefixes and embed documents in small batches."""

    def __init__(self, batch_size: int = 8) -> None:
        self.model = OllamaEmbeddings(model="nomic-embed-text")
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [
            f"search_document: {text}"
            for text in texts
        ]

        all_embeddings = []
        total = len(prefixed_texts)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = prefixed_texts[start:end]

            print(f"Embedding chunks {start + 1}-{end} of {total}...")

            batch_embeddings = self.model.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(f"search_query: {text}")
