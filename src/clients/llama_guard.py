"""
Llama Guard client adapter.
"""

from __future__ import annotations

import logging
import os
import re
import string
from typing import Optional, List

from ..utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

    LLAMA_GUARD_AVAILABLE = True
except ImportError:
    LLAMA_GUARD_AVAILABLE = False


class LlamaGuardClient:
    """Adapter for Meta's Llama Guard safety classifier."""

    MODEL_NAME = "meta-llama/LlamaGuard-7b"

    CATEGORY_MAPPING = {
        "S1": ["hate_violence"],
        "S2": ["illegal"],
        "S3": ["sexual"],
        "S4": ["sexual", "illegal"],
        "S5": ["illegal"],
        "S6": ["hate_violence"],
        "S7": ["hate_violence", "ideological"],
        "S8": ["self_inflicted"],
        "S9": ["sexual"],
        "S10": ["ideological"],
        "S11": ["illegal"],
    }

    PROMPT_TEMPLATE = string.Template(
        """<s>[INST] Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Specialized Advice.
S6: Indiscriminate Weapons.
S7: Hate.
S8: Self-Harm.
S9: Sexual Content.
S10: Elections.
S11: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: $text
<END CONVERSATION>

Provide your safety assessment for User in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
    )

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        if not LLAMA_GUARD_AVAILABLE:
            raise RuntimeError("Llama Guard requires: pip install torch transformers accelerate")

        if device is None:
            if torch.cuda.is_available():  # type: ignore
                device = "cuda"
            elif torch.backends.mps.is_available():  # type: ignore
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        logger.info("Loading Llama Guard from %s...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)  # type: ignore
        self.model = AutoModelForCausalLM.from_pretrained(  # type: ignore
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # type: ignore
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)  # type: ignore
        self.model.eval()
        logger.info("Llama Guard loaded on %s", self.device)

    def classify_text(self, text: str) -> tuple[bool, List[str]]:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty or whitespace-only")

        prompt = self.PROMPT_TEMPLATE.substitute(text=text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():  # type: ignore
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :],
            skip_special_tokens=True,
        ).strip()

        lines = response.split("\n")
        if not lines:
            logger.warning("Empty response from Llama Guard")
            return False, []

        first_line = lines[0].strip()
        first_token = re.split(r"\s+", first_line)[0].lower().strip(string.punctuation)
        is_unsafe = first_token.startswith("unsafe")

        violated_categories: List[str] = []

        if ":" in first_line or "-" in first_line:
            parts = re.split(r"[:\-]\s*", first_line, 1)
            if len(parts) > 1 and parts[1].strip():
                violated_categories.extend([c for c in re.split(r"[,\s]+", parts[1].strip()) if c])

        if len(lines) > 1 and lines[1].strip():
            violated_categories.extend([c for c in re.split(r"[,\s]+", lines[1].strip()) if c])

        violated_categories = list(set(violated_categories))

        if is_unsafe and not violated_categories:
            logger.warning("Model returned 'unsafe' without category metadata")

        return is_unsafe, violated_categories

    def categories_to_label(self, violated_categories: List[str]) -> HarmLabel:
        label = HarmLabel()
        if not violated_categories:
            return label

        harm_flags = {
            "hate_violence": False,
            "ideological": False,
            "sexual": False,
            "illegal": False,
            "self_inflicted": False,
        }

        for category in violated_categories:
            if category in self.CATEGORY_MAPPING:
                for harm_type in self.CATEGORY_MAPPING[category]:
                    harm_flags[harm_type] = True
            else:
                logger.warning(
                    "Unknown Llama Guard category '%s' not in CATEGORY_MAPPING. Full list: %s",
                    category,
                    violated_categories,
                )

        if harm_flags["hate_violence"]:
            label.hate_violence = Dimension.TOXIC
        if harm_flags["ideological"]:
            label.ideological = Dimension.TOXIC
        if harm_flags["sexual"]:
            label.sexual = Dimension.TOXIC
        if harm_flags["illegal"]:
            label.illegal = Dimension.TOXIC
        if harm_flags["self_inflicted"]:
            label.self_inflicted = Dimension.TOXIC

        return label

    def predict(self, text: str) -> HarmLabel:
        try:
            is_unsafe, violated_categories = self.classify_text(text)
            if violated_categories:
                return self.categories_to_label(violated_categories)
            if is_unsafe:
                return HarmLabel()
            return HarmLabel()
        except Exception as e:
            logger.error("Error classifying text: %s", e, exc_info=True)
            return HarmLabel()

    def __call__(self, text: str) -> HarmLabel:
        return self.predict(text)


# Backwards compatibility
LlamaGuard = LlamaGuardClient

