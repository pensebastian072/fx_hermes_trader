"""Resolve the candidate pair list against the real HF dataset catalog.

elthariel/histdata_fx_1m lays each symbol out as a top-level directory
(e.g. eurusd/ticks.parquet). We list the repo's files once, extract the
distinct top-level directory names, and case-insensitively match the
candidate majors+crosses list against them. Some thin crosses may not exist
in the 159-symbol set -- resolve honestly, don't force a substitute.

CLI: .venv\\Scripts\\python.exe -m gpu_meanrev.data.hf_catalog
"""
from __future__ import annotations

import json

import truststore
truststore.inject_into_ssl()

from gpu_meanrev import config


def list_catalog_symbols() -> list[str]:
    """Every top-level symbol directory in the HF dataset (uppercased)."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=config.HF_REPO_ID, repo_type=config.HF_REPO_TYPE)
    symbols = set()
    for f in files:
        if "/" in f:
            top = f.split("/", 1)[0]
            symbols.add(top.upper())
    return sorted(symbols)


def resolve_pairs(candidates: list[str] | None = None) -> dict:
    """Match candidates against the real catalog. Returns {resolved, missing}."""
    candidates = candidates or config.PAIRS_CANDIDATE
    catalog = set(list_catalog_symbols())
    resolved = [p for p in candidates if p.upper() in catalog]
    missing = [p for p in candidates if p.upper() not in catalog]
    return {"resolved": resolved, "missing": missing, "catalog_size": len(catalog)}


if __name__ == "__main__":
    result = resolve_pairs()
    print(json.dumps(result, indent=2))
    print(f"\n{len(result['resolved'])} resolved, {len(result['missing'])} missing "
          f"(catalog has {result['catalog_size']} symbols total)")
