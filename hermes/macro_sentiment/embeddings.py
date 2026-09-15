"""Embedding retrieval over the local central-bank corpus (upgrade from TF-IDF).

Embeddings come from a local Ollama instance (/api/embed, default model
nomic-embed-text) - local HTTP only, no API keys. Chunk embeddings are cached
on disk keyed by content hash so the corpus is only re-embedded when it
changes. If Ollama is unreachable the factory falls back to TfidfRetriever;
retrieval must never block on an LLM stack being up.

Both retrievers expose the same retrieve(query, top_k) -> list[Chunk]
interface, so callers don't care which one they got.
"""

import hashlib
import json
from pathlib import Path

import httpx
import numpy as np

from app.paths import data_dir
from hermes.macro_sentiment.retrieval import Chunk, TfidfRetriever, _chunk

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"
CACHE_NAME = "embedding_cache.json"


class OllamaEmbedder:
    def __init__(self, base_url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embeddings for each text, or None if Ollama is unreachable."""
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if not embeddings or len(embeddings) != len(texts):
                return None
            return embeddings
        except (httpx.HTTPError, json.JSONDecodeError):
            return None


class EmbeddingRetriever:
    """Cosine similarity over Ollama embeddings; same interface as TfidfRetriever."""

    def __init__(self, corpus_dir: Path, embedder: OllamaEmbedder | None = None,
                 cache_dir: Path | None = None):
        self.embedder = embedder or OllamaEmbedder()
        self.chunks: list[Chunk] = []
        for path in sorted(Path(corpus_dir).glob("*.txt")):
            for piece in _chunk(path.read_text(encoding="utf-8", errors="replace")):
                self.chunks.append(Chunk(source=path.name, text=piece))

        self._matrix: np.ndarray | None = None
        if self.chunks:
            vectors = self._embed_chunks(cache_dir or (data_dir() / "artifacts"))
            if vectors is not None:
                matrix = np.asarray(vectors, dtype=float)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                self._matrix = matrix / np.maximum(norms, 1e-12)

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def _embed_chunks(self, cache_dir: Path) -> list[list[float]] | None:
        cache_path = cache_dir / CACHE_NAME
        cache: dict[str, list[float]] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache = {}

        keys = [
            hashlib.sha256(f"{self.embedder.model}:{c.text}".encode()).hexdigest()
            for c in self.chunks
        ]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            new_vectors = self.embedder.embed([self.chunks[i].text for i in missing])
            if new_vectors is None:
                return None
            for i, vec in zip(missing, new_vectors):
                cache[keys[i]] = vec
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return [cache[k] for k in keys]

    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        if not self.chunks or self._matrix is None:
            return []
        q = self.embedder.embed([query])
        if q is None:
            return []
        q_vec = np.asarray(q[0], dtype=float)
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-12)
        sims = self._matrix @ q_vec
        ranked = np.argsort(sims)[::-1][:top_k]
        return [
            Chunk(source=self.chunks[i].source, text=self.chunks[i].text, score=float(sims[i]))
            for i in ranked
            if sims[i] > 0
        ]


def get_retriever(corpus_dir: Path, prefer_embeddings: bool = True):
    """EmbeddingRetriever when Ollama is up, TfidfRetriever otherwise."""
    if prefer_embeddings:
        retriever = EmbeddingRetriever(corpus_dir)
        if retriever.ready:
            return retriever
    return TfidfRetriever(corpus_dir)
