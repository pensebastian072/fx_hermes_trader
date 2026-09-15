"""TF-IDF retrieval over local central-bank documents (RAG, local-only).

Corpus: plain-text files in data/raw/central_bank/. No external services,
no embeddings server required; swap in a local embedding model later behind
the same retrieve() interface.
"""

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_CHARS = 800


@dataclass
class Chunk:
    source: str
    text: str
    score: float = 0.0


def _chunk(text: str, size: int = CHUNK_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) > size and current:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}".strip()
    if current:
        chunks.append(current)
    return chunks


class TfidfRetriever:
    def __init__(self, corpus_dir: Path):
        self.chunks: list[Chunk] = []
        for path in sorted(Path(corpus_dir).glob("*.txt")):
            for piece in _chunk(path.read_text(encoding="utf-8", errors="replace")):
                self.chunks.append(Chunk(source=path.name, text=piece))
        self._vectorizer = None
        self._matrix = None
        if self.chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        if not self.chunks:
            return []
        sims = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        ranked = sims.argsort()[::-1][:top_k]
        return [
            Chunk(source=self.chunks[i].source, text=self.chunks[i].text, score=float(sims[i]))
            for i in ranked
            if sims[i] > 0
        ]
