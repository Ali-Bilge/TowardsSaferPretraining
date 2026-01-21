"""
OpenRouter-backed TTP client.

OpenRouter exposes an OpenAI-compatible API surface. We reuse the OpenAI SDK but
set the OpenRouter base URL and optional recommended headers.
"""

from __future__ import annotations

from typing import Optional, Dict

from openai import OpenAI  # type: ignore

from .ttp_openai import OpenAITTPClient


class OpenRouterTTPClient(OpenAITTPClient):
    """
    TTP client that sends Chat Completions to OpenRouter.

    - Base URL: https://openrouter.ai/api/v1
    - Key: OPENROUTER_API_KEY (or pass api_key)
    - Model IDs: OpenRouter-style (e.g. "openai/gpt-4o")
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o",
        prompt_path: str = "prompts/TTP/TTP.txt",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = DEFAULT_BASE_URL,
        referer: Optional[str] = None,
        title: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            prompt_path=prompt_path,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        headers: Dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        # Override the OpenAI client to point at OpenRouter.
        # (OpenRouter is OpenAI-compatible, so the request schema stays the same.)
        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)

