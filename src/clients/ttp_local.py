"""
Local (Transformers) TTP client.

Used for Table 4 rows where the paper runs the TTP prompt on non-OpenAI models
(e.g., Gemma 2 27B).

We reuse the exact ChatML prompt from `prompts/TTP/TTP.txt` and ask the model
to produce the same <Label>{...}</Label> structure, then parse it into HarmLabel.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Literal

from ..utils.taxonomy import HarmLabel

logger = logging.getLogger(__name__)


@dataclass
class LocalTTPResult:
    predicted_label: HarmLabel
    raw_response: Optional[str] = None
    error: Optional[str] = None


class TransformersTTPClient:
    """
    Run the TTP prompt on a local HuggingFace CausalLM.

    This is meant for replication, not production. Defaults to deterministic generation.
    """

    def __init__(
        self,
        model_id: str,
        *,
        prompt_path: str = "prompts/TTP/TTP.txt",
        device: Optional[str] = None,
        dtype: Literal["auto", "float16", "bfloat16"] = "auto",
        quantization: Literal["none", "8bit", "4bit"] = "none",
        # Keep this small: we only need a short <Label> dict, and larger values can OOM on 40GB GPUs.
        max_new_tokens: int = 128,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
        except ImportError as e:
            raise RuntimeError("Local TTP requires: pip install torch transformers accelerate") from e

        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        # Cap input length to avoid KV-cache OOM on large prompts.
        # Override via env if you want to force a smaller cap.
        self.max_input_tokens: Optional[int] = None
        env_max_in = os.environ.get("TTP_LOCAL_MAX_INPUT_TOKENS")
        if env_max_in and env_max_in.strip():
            try:
                self.max_input_tokens = int(env_max_in)
            except Exception:
                self.max_input_tokens = None

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"TTP prompt not found: {prompt_path}")
        self.prompt_template = prompt_file.read_text(encoding="utf-8", errors="replace")
        self._parse_prompt_template()

        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        auth_kwargs: Dict[str, Any] = {"token": hf_token} if hf_token else {}

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, **auth_kwargs)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        self.tokenizer = tok

        # dtype handling
        if dtype == "auto":
            # On CUDA, defaulting to fp32 is often too large for 27B/32B models.
            torch_dtype = torch.float16 if device == "cuda" else None
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.bfloat16

        # quantization handling (optional dependency)
        load_kwargs: Dict[str, Any] = {}
        if quantization in {"8bit", "4bit"}:
            try:
                import bitsandbytes  # noqa: F401  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "Requested quantization but bitsandbytes is not installed. "
                    "Install with: pip install bitsandbytes"
                ) from e
            if quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["load_in_4bit"] = True

        logger.info("Loading local TTP model %s on %s (quantization=%s)...", model_id, device, quantization)
        if device == "cuda" and torch.cuda.device_count() > 1:
            # Encourage sharding across multiple GPUs instead of filling GPU:0.
            # Snellius A100 nodes are often 40GB per GPU; leave some headroom for KV cache.
            load_kwargs.setdefault("max_memory", {i: "38GiB" for i in range(torch.cuda.device_count())})
            load_kwargs["max_memory"].setdefault("cpu", "64GiB")
            device_map = "balanced"
        else:
            device_map = "auto" if device == "cuda" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **auth_kwargs,
            **load_kwargs,
        )
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

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

        prefix = []
        for role, content in blocks[:template_idx]:
            role = role.strip().lower()
            content = content.strip()
            if not content:
                continue
            prefix.append((role, content))
        if not prefix or prefix[0][0] != "system":
            raise ValueError("Prompt did not start with a system message block")
        self._prefix = prefix

    def _format_prompt(self, user_message: str) -> str:
        # Prefer the model's chat template when available (important for instruct models like Gemma / LLaMA).
        if hasattr(self.tokenizer, "apply_chat_template"):
            # Try a couple role conventions: many models use "assistant", Gemma often uses "model".
            for map_assistant_to_model in (False, True):
                try:
                    messages = []
                    for role, content in self._prefix:
                        r = "model" if (map_assistant_to_model and role == "assistant") else role
                        messages.append({"role": r, "content": content})
                    messages.append({"role": "user", "content": user_message})
                    return self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    continue

        # Fallback: represent the ChatML conversation explicitly for a plain-text CausalLM.
        parts = []
        for role, content in self._prefix:
            parts.append(f"{role.upper()}:\n{content}")
        parts.append(f"USER:\n{user_message}")
        return "\n\n".join(parts)

    def _parse_response(self, content: str) -> HarmLabel:
        label_match = re.search(r"<Label>\s*({.*?})\s*</Label>", content, re.DOTALL)
        label_str: Optional[str] = None
        if label_match:
            label_str = label_match.group(1)
        else:
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
                label_dict = parsed
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
        return HarmLabel.from_dict(clean)

    def evaluate(self, url: str, body: str) -> LocalTTPResult:
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)
        # Local instruct models are prone to verbose reasoning; make the required output format explicit.
        user_message = (
            user_message
            + "\n\nIMPORTANT: Output ONLY a single label dict in the exact form "
            + "<Label>{H: <None|Topical-*>|Intent-*>, IH: <...>, SE: <...>, IL: <...>, SI: <...>}</Label> "
            + "and nothing else."
        )
        prompt = self._format_prompt(user_message)

        try:
            max_len = self.max_input_tokens
            if max_len is None:
                # tokenizer.model_max_length is sometimes a huge sentinel; clamp to something reasonable by default.
                tmax = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
                max_len = tmax if 0 < tmax <= 32768 else 8192
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(max_len))
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=8,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            try:
                lbl = self._parse_response(text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse TTP label. output_head={text[:240]!r}. error={e}") from e
            return LocalTTPResult(predicted_label=lbl, raw_response=text)
        except Exception as e:
            return LocalTTPResult(predicted_label=HarmLabel(), raw_response=None, error=str(e))

    def predict(self, text: str) -> HarmLabel:
        r = self.evaluate(url="ttp://text", body=text)
        # Do not silently fail-open: callers (e.g. evaluation scripts) should count failures.
        if r.error:
            raise RuntimeError(r.error)
        return r.predicted_label

