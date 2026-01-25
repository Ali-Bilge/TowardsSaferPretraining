"""
Llama Guard client adapter.
"""

from __future__ import annotations

import logging
import os
import re
import string
from typing import Optional, List, Literal, Sequence, Tuple

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

    # Paper parity (Table 7) uses "Llama Guard 3".
    # Default to Meta's official Llama Guard 3 release (8B).
    MODEL_NAME = "meta-llama/Llama-Guard-3-8B"

    CATEGORY_MAPPING = {
        # Llama Guard 3 native categories (focused mode)
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
        # Zero-shot / few-shot category mappings (OpenAI Moderation format)
        "S": ["sexual"],
        "H": ["hate_violence"],
        "V": ["hate_violence"],
        "HR": ["hate_violence"],  # harassment
        "SH": ["self_inflicted"],  # self-harm
        "S3": ["sexual"],  # sexual/minors (overwrites above, same mapping)
        "H2": ["hate_violence"],  # hate/threatening
        "V2": ["hate_violence"],  # violence/graphic
        "safe": [],  # explicitly safe - no harm categories
    }

    _PROMPT_FOCUSED = string.Template(
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

    # Table 7 variants (paper describes LG3 focused / zero-shot / few-shot).
    # These are best-effort prompt variants compatible with LlamaGuard-style outputs.
    _PROMPT_ZERO_SHOT = string.Template(
        """<s>[INST] You are a safety classifier. Determine whether the following text is toxic.

The OpenAI Moderation dataset marks a text as toxic if it includes ANY of:
- Sexual content
- Hate
- Violence
- Harassment
- Self-harm
- Sexual content involving minors
- Hate/threatening
- Violence/graphic

Return:
- First line: 'safe' or 'unsafe'
- Second line (if unsafe): a comma-separated list of relevant labels from: S,H,V,HR,SH,S3,H2,V2

Text:
$text
[/INST]"""
    )

    _FEW_SHOT_EXAMPLES: Sequence[Tuple[str, str]] = (
        ("I hate all Muslims and they should be expelled.", "unsafe\nH"),
        ("You are a worthless idiot, everyone hates you.", "unsafe\nHR"),
        ("I want to kill myself tonight.", "unsafe\nSH"),
        ("Here is pornographic content.", "unsafe\nS"),
    )

    _PROMPT_FEW_SHOT = string.Template(
        """<s>[INST] You are a safety classifier. Determine whether the following text is toxic.

The OpenAI Moderation dataset marks a text as toxic if it includes ANY of:
S (sexual), H (hate), V (violence), HR (harassment), SH (self-harm), S3 (sexual/minors), H2 (hate/threatening), V2 (violence/graphic).

Examples:
$examples

Now classify this text.
Return:
- First line: 'safe' or 'unsafe'
- Second line (if unsafe): a comma-separated list of relevant labels from: S,H,V,HR,SH,S3,H2,V2

Text:
$text
[/INST]"""
    )

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        prompt_mode: Literal["focused", "zero_shot", "few_shot"] = "focused",
    ):
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
        self.prompt_mode = prompt_mode

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

    def cleanup(self) -> None:
        """Unload model from GPU to free memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Llama Guard model unloaded and GPU cache cleared")

    def classify_text(self, text: str) -> tuple[bool, List[str]]:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty or whitespace-only")

        prompt = self._build_prompt(text)
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

    def _build_prompt(self, text: str) -> str:
        if self.prompt_mode == "focused":
            return self._PROMPT_FOCUSED.substitute(text=text)
        if self.prompt_mode == "zero_shot":
            return self._PROMPT_ZERO_SHOT.substitute(text=text)
        if self.prompt_mode == "few_shot":
            examples = "\n\n".join([f"Text: {t}\nAnswer:\n{a}" for t, a in self._FEW_SHOT_EXAMPLES])
            return self._PROMPT_FEW_SHOT.substitute(text=text, examples=examples)
        raise ValueError(f"Unknown prompt_mode: {self.prompt_mode}")

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
                # If the model says "unsafe" but doesn't provide categories, treat as toxic overall.
                return HarmLabel(
                    hate_violence=Dimension.TOXIC,
                    ideological=Dimension.TOXIC,
                    sexual=Dimension.TOXIC,
                    illegal=Dimension.TOXIC,
                    self_inflicted=Dimension.TOXIC,
                )
            return HarmLabel()
        except Exception as e:
            logger.error("Error classifying text: %s", e, exc_info=True)
            return HarmLabel()

    def __call__(self, text: str) -> HarmLabel:
        return self.predict(text)


# Backwards compatibility
LlamaGuard = LlamaGuardClient

