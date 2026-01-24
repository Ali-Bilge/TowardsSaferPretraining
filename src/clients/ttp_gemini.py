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
        blocks = re.findall(
            r"<\|im_start\|>(system|user|assistant)\s*(.*?)<\|im_end\|>",
            self.prompt_template,
            re.DOTALL,
        )
        if not blocks:
            raise ValueError("Could not parse any ChatML blocks from prompt")

        template_idx: Optional[int] = None
        for i in range(len(blocks) - 1, -1, -1):
            role, content = blocks[i]
            if role == "user" and "#URL#" in content and "#Body#" in content:
                template_idx = i
                break
        if template_idx is None:
            raise ValueError("Could not find final user template block containing #URL# and #Body#")

        self.user_template = blocks[template_idx][1].strip()

        prefix_msgs = []
        for role, content in blocks[:template_idx]:
            content = content.strip()
            if not content:
                continue
            prefix_msgs.append((role.strip(), content))
        if not prefix_msgs or prefix_msgs[0][0] != "system":
            raise ValueError("Prompt did not start with a system message block")

        self._prefix_messages = prefix_msgs
        self.system_message = prefix_msgs[0][1]

    def _format_prompt(self, final_user_message: str) -> str:
        """
        Gemini SDKs accept a single string prompt in our current usage; represent the
        ChatML conversation explicitly with role tags.
        """
        parts = []
        for role, content in self._prefix_messages:
            parts.append(f"{role.upper()}:\n{content}")
        parts.append(f"USER:\n{final_user_message}")
        return "\n\n".join(parts)

    def evaluate(self, url: str, body: str) -> TTPResult:
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)
        # Gemini also needs explicit output format instructions
        user_message = (
            user_message
            + "\n\n# OUTPUT FORMAT\n"
            + "You must respond with ONLY the label in this exact format:\n"
            + "<Label>{H: None|Topical-i|Intent-i, IH: None|Topical-i|Intent-i, SE: None|Topical-i|Intent-i, IL: None|Topical-i|Intent-i, SI: None|Topical-i|Intent-i}</Label>\n"
            + "Do not include any other text, reasoning, or explanation."
        )

        last_text: Optional[str] = None
        for attempt in range(self.max_retries):
            try:
                if self._sdk == "google-genai":
                    prompt = self._format_prompt(user_message)
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
                        self._format_prompt(user_message),
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
                # If the project has 0 quota, retrying just wastes time.
                if "RESOURCE_EXHAUSTED" in msg and "limit: 0" in msg:
                    return TTPResult(
                        url=url,
                        body=body,
                        predicted_label=HarmLabel(),
                        raw_response=last_text,
                        error=msg,
                    )
                try:
                    m = re.search(r"(?:retry after|retry in)\s+([0-9]+\.?[0-9]*)s", msg, re.IGNORECASE)
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
        # Do not silently fail-open: callers (e.g. evaluation scripts) should count failures.
        if result.error:
            raise RuntimeError(result.error)
        return result.predicted_label

    def _parse_response(self, content: str) -> tuple[HarmLabel, Optional[str]]:
        reasoning_match = re.search(r"<Reasoning>(.*?)</Reasoning>", content, re.DOTALL | re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        content = content.strip()

        # First try: exact <Label> tag format
        label_match = re.search(r"<Label>\s*({.*?})\s*</Label>", content, re.DOTALL | re.IGNORECASE)
        label_str: Optional[str] = None
        if label_match:
            label_str = label_match.group(1)
        else:
            # Second try: look for any JSON-like dict containing the required keys
            for m in re.finditer(r"\{[\s\S]*?\}", content):
                candidate = m.group(0)
                if all(k in candidate for k in ["H", "IH", "SE", "IL", "SI"]):
                    label_str = candidate
                    break

            # Third try: if content starts with a brace and contains the keys
            if not label_str and content.startswith("{") and all(k in content for k in ["H", "IH", "SE", "IL", "SI"]):
                # Find the closing brace
                brace_count = 0
                end_pos = 0
                for i, char in enumerate(content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                if end_pos > 0:
                    label_str = content[:end_pos]

        if not label_str:
            # Last resort: try to extract from natural language response
            # Look for patterns like "H: None", "IH: Intent-1", etc.
            extracted = {}
            for key in ["H", "IH", "SE", "IL", "SI"]:
                pattern = rf"{key}\s*:\s*([^{{}}\n,]*)"
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"\']')
                    if value:
                        extracted[key] = value.split("-")[0].lower()
            if extracted and len(extracted) >= 3:  # At least 3 keys found
                label_str = str(extracted).replace("'", '"')

        if not label_str:
            raise ValueError(f"Could not find label dict in response: {content[:300]}")

        try:
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, dict):
                label_dict = parsed
            elif isinstance(parsed, str):
                label_dict = json.loads(parsed)
            else:
                raise ValueError(f"Expected dict or string from literal_eval, got {type(parsed)}")
        except (ValueError, SyntaxError, json.JSONDecodeError):
            # Try to fix common issues
            label_str = label_str.replace("'", '"')  # Replace single quotes with double
            try:
                label_dict = json.loads(label_str)
            except json.JSONDecodeError:
                # Try to parse individual key-value pairs
                label_dict = {}
                for key in ["H", "IH", "SE", "IL", "SI"]:
                    match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', label_str)
                    if match:
                        value = match.group(1).split("-")[0].lower()
                        label_dict[key] = value
                    else:
                        match = re.search(rf"{key}\s*:\s*([^,}}]+)", label_str)
                        if match:
                            value = match.group(1).strip().strip('"\']').split("-")[0].lower()
                            label_dict[key] = value

        if not label_dict:
            raise ValueError(f"Failed to parse label dict from: {label_str}")

        clean: Dict[str, Any] = {}
        for k, v in label_dict.items():
            if isinstance(v, str):
                clean[k] = v.split("-")[0].lower()
            else:
                clean[k] = str(v).lower()
        return HarmLabel.from_dict(clean), reasoning


# Backwards compatibility
GeminiTTPEvaluator = GeminiTTPClient

