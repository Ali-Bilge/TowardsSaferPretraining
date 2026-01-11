"""
Llama Guard wrapper for content safety classification.

Uses Meta's Llama Guard model for toxicity detection.
"""

import logging
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


class LlamaGuard:
    """
    Wrapper for Meta's Llama Guard safety classifier.

    Llama Guard categories:
    - S1: Violent Crimes
    - S2: Non-Violent Crimes
    - S3: Sex Crimes
    - S4: Child Exploitation
    - S5: Specialized Advice (defamation, privacy, etc.)
    - S6: Indiscriminate Weapons
    - S7: Hate
    - S8: Self-Harm
    - S9: Sexual Content
    - S10: Elections
    - S11: Code Interpreter Abuse

    Maps to our taxonomy:
    - S1, S6, S7 -> Hate/Violence (H)
    - S7, S10 -> Ideological Harm (IH)
    - S3, S4, S9 -> Sexual Content (SE)
    - S2, S5 -> Illegal Activities (IL)
    - S8 -> Self-Inflicted Harm (SI)

    Example:
        guard = LlamaGuard()
        label = guard.predict("This is a test text")
    """

    # Model name on HuggingFace
    MODEL_NAME = "meta-llama/LlamaGuard-7b"

    # Mapping from Llama Guard categories to our taxonomy
    CATEGORY_MAPPING = {
        "S1": ["hate_violence"],          # Violent Crimes
        "S2": ["illegal"],                # Non-Violent Crimes
        "S3": ["sexual"],                 # Sex Crimes
        "S4": ["sexual", "illegal"],       # Child Exploitation
        "S5": ["illegal"],                # Specialized Advice
        "S6": ["hate_violence"],          # Indiscriminate Weapons
        "S7": ["hate_violence", "ideological"],  # Hate
        "S8": ["self_inflicted"],         # Self-Harm
        "S9": ["sexual"],                 # Sexual Content
        "S10": ["ideological"],           # Elections
        "S11": ["illegal"],               # Code Interpreter Abuse
    }

    # Prompt template for Llama Guard
    PROMPT_TEMPLATE = string.Template("""<s>[INST] Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

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
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""")

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None
    ):
        """
        Initialize Llama Guard classifier.

        Args:
            model_name: HuggingFace model name
            device: Device (cuda/cpu/mps)
        """
        if not LLAMA_GUARD_AVAILABLE:
            raise RuntimeError(
                "Llama Guard requires: pip install torch transformers accelerate"
            )

        # Determine target device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        logger.info(f"Loading Llama Guard from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        # Move model to device only when device_map is not used (device_map handles placement automatically)
        if self.device != "cuda":
            self.model.to(self.device)

        self.model.eval()
        logger.info(f"Llama Guard loaded on {self.device}")

    def classify_text(self, text: str) -> tuple[bool, List[str]]:
        """
        Classify text with Llama Guard.

        Args:
            text: Text to classify

        Returns:
            Tuple of (is_unsafe, violated_categories)
        """
        # Validate input text
        text = text.strip()
        if not text:
            raise ValueError("text cannot be empty or whitespace-only")

        # Format prompt
        prompt = self.PROMPT_TEMPLATE.substitute(text=text)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Deterministic
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        # Parse response
        lines = response.split('\n')
        if not lines:
            logger.warning("Empty response from Llama Guard")
            return False, []

        # Normalize first line and extract safety label
        first_line = lines[0].strip()
        # Extract first meaningful token, clean it up
        first_token = re.split(r'\s+', first_line)[0].lower().strip(string.punctuation)
        # Check for unsafe variants using startswith (handles "unsafe:", "unsafe -", "unsafe (high)", etc.)
        is_unsafe = first_token.startswith("unsafe")

        violated_categories = []

        # Parse categories from first line after separator
        if ':' in first_line or '-' in first_line:
            separator_pattern = r'[:\-]\s*'
            parts = re.split(separator_pattern, first_line, 1)
            if len(parts) > 1:
                category_part = parts[1].strip()
                if category_part:
                    # Split on commas and other delimiters
                    categories = re.split(r'[,\s]+', category_part)
                    violated_categories.extend([c.strip() for c in categories if c.strip()])

        # Parse categories from second line if present
        if len(lines) > 1:
            categories_line = lines[1].strip()
            if categories_line:
                # Split on commas and other delimiters
                categories = re.split(r'[,\s]+', categories_line)
                violated_categories.extend([c.strip() for c in categories if c.strip()])

        # Filter out empty entries and duplicates
        violated_categories = list(set(violated_categories))

        if is_unsafe and not violated_categories:
            logger.warning("Model returned 'unsafe' without category metadata")

        return is_unsafe, violated_categories

    def categories_to_label(
        self,
        violated_categories: List[str]
    ) -> 'HarmLabel':
        """
        Convert Llama Guard categories to HarmLabel.

        Args:
            violated_categories: List of violated category codes (e.g., ["S1", "S7"])

        Returns:
            HarmLabel with classifications
        """
        label = HarmLabel()

        if not violated_categories:
            return label  # All safe

        # Map violated categories to harm dimensions
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
                    f"Unknown Llama Guard category '{category}' not in CATEGORY_MAPPING. "
                    f"Full violated_categories: {violated_categories}"
                )

        # Set dimensions (Llama Guard only does binary toxic/safe, no topical)
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

    def predict(self, text: str) -> 'HarmLabel':
        """
        Predict harm label for text.

        Args:
            text: Text to classify

        Returns:
            HarmLabel with classifications
        """
        try:
            is_unsafe, violated_categories = self.classify_text(text)
            if violated_categories:
                return self.categories_to_label(violated_categories)
            elif is_unsafe:
                # Model detected unsafe content but didn't specify categories
                # Set all dimensions to TOXIC as conservative approach for unspecified toxicity
                label = HarmLabel()
                label.hate_violence = Dimension.TOXIC
                label.ideological = Dimension.TOXIC
                label.sexual = Dimension.TOXIC
                label.illegal = Dimension.TOXIC
                label.self_inflicted = Dimension.TOXIC
                return label
            else:
                return HarmLabel()  # All safe
        except Exception as e:
            logger.error(f"Error classifying text: {e}", exc_info=True)
            return HarmLabel()

    def __call__(self, text: str) -> 'HarmLabel':
        """Make classifier callable."""
        return self.predict(text)
