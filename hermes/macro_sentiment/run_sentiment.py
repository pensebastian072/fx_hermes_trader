"""Batch sentiment update: retrieve relevant chunks from the local central-bank
corpus, run the multi-agent debate, write the consensus artifact.

Usage:
  python -m hermes.macro_sentiment.run_sentiment --currency USD --query "policy rate inflation"
  add --inflation-above-target / --inflation-below-target for context
  add --ollama to use a local Ollama model for agent reasoning

Output: data/artifacts/sentiment_<currency>.json (bias only, never a trade).
"""

import argparse

from app.paths import data_dir
from hermes.macro_sentiment.debate import run_debate
from hermes.macro_sentiment.embeddings import get_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--query", default="inflation policy rate outlook")
    parser.add_argument("--inflation-above-target", action="store_true", dest="above")
    parser.add_argument("--inflation-below-target", action="store_true", dest="below")
    parser.add_argument("--ollama", action="store_true", help="use local Ollama agents")
    parser.add_argument(
        "--tfidf", action="store_true", help="force TF-IDF retrieval (skip embeddings)"
    )
    args = parser.parse_args()

    corpus = data_dir() / "raw" / "central_bank"
    retriever = get_retriever(corpus, prefer_embeddings=not args.tfidf)
    print(f"retriever: {type(retriever).__name__}")
    chunks = retriever.retrieve(args.query, top_k=4)
    if not chunks:
        print(f"No documents in {corpus}. Drop .txt transcripts there first.")
        return

    text = "\n\n".join(c.text for c in chunks)
    inflation_above = True if args.above else (False if args.below else None)

    llm_client = None
    if args.ollama:
        from hermes.macro_sentiment.ollama_client import OllamaClient

        llm_client = OllamaClient()

    consensus = run_debate(
        text, currency=args.currency, inflation_above_target=inflation_above,
        llm_client=llm_client,
    )
    artifacts = data_dir() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    path = artifacts / f"sentiment_{args.currency.lower()}.json"
    path.write_text(consensus.model_dump_json(indent=2), encoding="utf-8")
    print(f"{args.currency}: {consensus.macro_bias} ({consensus.hawkish_dovish_score:+.1f}), "
          f"confidence {consensus.confidence:.2f} -> {path}")


if __name__ == "__main__":
    main()
