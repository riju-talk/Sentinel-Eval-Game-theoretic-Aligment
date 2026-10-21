"""Minimal LLM client layer: Anthropic, any OpenAI-compatible server (OpenAI, Ollama, vLLM), and a mock."""
from __future__ import annotations

import os
from typing import Callable, Protocol

import httpx

Message = dict  # {"role": "user"|"assistant", "content": str}


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[Message], temperature: float = 0.0,
                 max_tokens: int = 1024) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None, timeout: float = 120.0):
        self.name = f"anthropic:{model}"
        self.model = model
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.timeout = timeout

    def complete(self, system, messages, temperature=0.0, max_tokens=1024):
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            json={"model": self.model, "system": system, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class OpenAICompatLLM:
    """Works with OpenAI, Ollama (http://localhost:11434/v1), vLLM, LM Studio."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, timeout: float = 300.0):
        self.name = f"openai:{model}@{base_url}"
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")

    def complete(self, system, messages, temperature=0.0, max_tokens=1024):
        r = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system}, *messages]},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class MockLLM:
    """Deterministic LLM for tests and offline runs. `fn(system, messages) -> str`."""

    def __init__(self, fn: Callable[[str, list[Message]], str], name: str = "mock"):
        self.fn, self.name = fn, name
        self.calls = 0

    def complete(self, system, messages, temperature=0.0, max_tokens=1024):
        self.calls += 1
        return self.fn(system, messages)


def make_llm(spec: str) -> LLM:
    """`anthropic:<model>` | `ollama:<model>` | `openai:<model>` | `openai:<model>@<base_url>`."""
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return AnthropicLLM(rest)
    if kind == "ollama":
        return OpenAICompatLLM(rest, base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"))
    if kind == "openai":
        model, _, url = rest.partition("@")
        return OpenAICompatLLM(model, base_url=url or "https://api.openai.com/v1")
    raise ValueError(f"unknown LLM spec {spec!r}")
