"""Embedding retrieval: stubbed embedder, disk cache, TF-IDF fallback."""

import json

import numpy as np

from hermes.macro_sentiment.embeddings import (
    CACHE_NAME,
    EmbeddingRetriever,
    OllamaEmbedder,
    get_retriever,
)
from hermes.macro_sentiment.retrieval import TfidfRetriever


class StubEmbedder:
    """Deterministic 'embeddings': bag of letter counts. No network."""

    model = "stub"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [
            [float(t.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"]
            for t in texts
        ]


class DeadEmbedder:
    model = "dead"

    def embed(self, texts):
        return None


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "fed.txt").write_text(
        "inflation remains elevated and the committee will raise rates",
        encoding="utf-8",
    )
    (corpus / "boj.txt").write_text(
        "yield curve control unchanged, accommodative stance continues",
        encoding="utf-8",
    )
    return corpus


def test_retrieve_ranks_by_similarity(data_dir, tmp_path):
    corpus = _corpus(tmp_path)
    retriever = EmbeddingRetriever(corpus, embedder=StubEmbedder())
    assert retriever.ready
    results = retriever.retrieve("inflation rates committee", top_k=1)
    assert results and results[0].source == "fed.txt"
    assert results[0].score > 0


def test_cache_prevents_reembedding(data_dir, tmp_path):
    corpus = _corpus(tmp_path)
    cache_dir = tmp_path / "cache"

    first = StubEmbedder()
    EmbeddingRetriever(corpus, embedder=first, cache_dir=cache_dir)
    assert first.calls == 1
    cache = json.loads((cache_dir / CACHE_NAME).read_text(encoding="utf-8"))
    assert len(cache) == 2

    second = StubEmbedder()
    EmbeddingRetriever(corpus, embedder=second, cache_dir=cache_dir)
    assert second.calls == 0  # all chunks served from cache


def test_dead_embedder_marks_not_ready(data_dir, tmp_path):
    retriever = EmbeddingRetriever(_corpus(tmp_path), embedder=DeadEmbedder())
    assert not retriever.ready
    assert retriever.retrieve("anything") == []


def test_get_retriever_falls_back_to_tfidf(data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(OllamaEmbedder, "embed", lambda self, texts: None)
    retriever = get_retriever(_corpus(tmp_path))
    assert isinstance(retriever, TfidfRetriever)
    assert retriever.retrieve("inflation rates", top_k=1)[0].source == "fed.txt"
