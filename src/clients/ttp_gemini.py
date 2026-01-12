"""
Gemini-backed TTP client.

It's a client/adapter: it produces labels by calling an API.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict

from ..utils.taxonomy import HarmLabel

logger = logging.getLogger(__name__)


@dataclass
class TTPResult:
    """Result from TTP evaluation (Gemini)."""

    url: str
    body: str
    predicted_label: HarmLabel
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None


class GeminiTTPClient:
    """
    Client using the TTP prompt with Gemini (Google AI Studio).

    Environment variables supported:
    - GEMINI_API_KEY (or pass api_key)
    - GEMINI_MODEL (default: gemini-2.0-flash)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        prompt_path: str = "prompts/TTP/TTP.txt",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key missing. Set GEMINI_API_KEY or pass api_key=...")

        self.model = model or os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash"
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"TTP prompt not found: {prompt_path}")
        self.prompt_template = prompt_file.read_text(encoding="utf-8")
        self._parse_prompt_template()

        # Lazy import: prefer `google-genai` SDK, fallback to deprecated `google-generativeai`.
        try:
            import google.genai as genai  # type: ignore

            self._sdk = "google-genai"
            self._client = genai.Client(api_key=self.api_key)  # type: ignore[attr-defined]
        except Exception:
            try:
                import google.generativeai as genai  # type: ignore

                self._sdk = "google-generativeai"
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(  # type: ignore[attr-defined]
                    model_name=self.model,
                    system_instruction=self.system_message,
                )
            except Exception as e:
                raise RuntimeError(
                    "Gemini support requires either google-genai (preferred) or google-generativeai.\n"
                    "Install with: pip install -U google-genai"
                ) from e

    def _parse_prompt_template(self) -> None:
        system_match = re.search(
            r"<\|im_start\|>system\s*(.*?)<\|im_end\|>",
            self.prompt_template,
            re.DOTALL,
        )
        if not system_match:
            raise ValueError("Could not find system message in prompt")
        self.system_message = system_match.group(1).strip()

        user_blocks = re.findall(
            r"<\|im_start\|>user\s*(.*?)<\|im_end\|>",
            self.prompt_template,
            re.DOTALL,
        )
        chosen: Optional[str] = None
        if user_blocks:
            for blk in reversed(user_blocks):
                if "#URL#" in blk and "#Body#" in blk:
                    chosen = blk.strip()
                    break
            if chosen is None:
                chosen = user_blocks[-1].strip()

        if not chosen:
            raise ValueError("Could not find user message template in prompt")
        self.user_template = chosen

    def evaluate(self, url: str, body: str) -> TTPResult:
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)

        last_text: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                if self._sdk == "google-genai":
                    # google-genai: pass full content including system message as a single prompt.
                    prompt = f"{self.system_message}\n\n{user_message}"
                    resp = self._client.models.generate_content(  # type: ignore[attr-defined]
                        model=self.model,
                        contents=prompt,
                        config={"temperature": self.temperature},
                    )
                    text = getattr(resp, "text", None)
                    if text is None:
                        raise ValueError("Gemini response missing text")
                else:
                    resp = self._model.generate_content(  # type: ignore[attr-defined]
                        user_message,
                        generation_config={"temperature": self.temperature},
                    )
                    text = getattr(resp, "text", None)
                    if text is None:
                        raise ValueError("Gemini response missing text")

                last_text = text
                label, reasoning = self._parse_response(text)
                return TTPResult(
                    url=url,
                    body=body,
                    predicted_label=label,
                    reasoning=reasoning,
                    raw_response=text,
                )
            except Exception as e:
                # Attempt to respect server-provided backoff if present.
                msg = str(e)
                try:
                    m = re.search(r"retry after\s+(\d+)s", msg, re.IGNORECASE)
                    if m:
                        time.sleep(max(float(m.group(1)), self.retry_delay))
                        continue
                except Exception:
                    pass

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return TTPResult(
                        url=url,
                        body=body,
                        predicted_label=HarmLabel(),
                        raw_response=last_text,
                        error=str(e),
                    )

        raise RuntimeError("Unexpected execution path in evaluate method")

    def predict(self, text: str) -> HarmLabel:
        result = self.evaluate(url="ttp://text", body=text)
        return result.predicted_label

    def _parse_response(self, content: str) -> tuple[HarmLabel, Optional[str]]:
        reasoning_match = re.search(r"<Reasoning>(.*?)</Reasoning>", content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        label_match = re.search(r"<Label>\s*({.*?})\s*</Label>", content, re.DOTALL)
        label_str: Optional[str] = label_match.group(1) if label_match else None

        if not label_str:
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                label_str = stripped
            else:
                for m in re.finditer(r"\{[\s\S]*?\}", content):
                    candidate = m.group(0)
                    if all(k in candidate for k in ["H", "IH", "SE", "IL", "SI"]):
                        label_str = candidate
                        break

        if not label_str:
            raise ValueError(f"Could not find label dict in response: {content[:200]}")

        try:
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, dict):
                label_dict: Dict[str, Any] = parsed
            elif isinstance(parsed, str):
                label_dict = json.loads(parsed)
            else:
                raise ValueError(f"Expected dict or string from literal_eval, got {type(parsed)}")
        except (ValueError, SyntaxError):
            label_dict = json.loads(label_str)

        clean: Dict[str, Any] = {}
        for k, v in label_dict.items():
            if isinstance(v, str):
                clean[k] = v.split("-")[0].lower()
            else:
                clean[k] = v
        return HarmLabel.from_dict(clean), reasoning


# Backwards compatibility
GeminiTTPEvaluator = GeminiTTPClient

