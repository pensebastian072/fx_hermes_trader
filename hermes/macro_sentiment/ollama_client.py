"""Thin client for a local Ollama instance. Local HTTP only, no API keys.

If Ollama is not running, callers fall back to the deterministic lexicon
model - sentiment must never block on an LLM.
"""

import json

import httpx

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_URL, model: str = "hermes3:8b", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str | None:
        """Return the model response, or None if Ollama is unreachable."""
        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("response")
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            return None
