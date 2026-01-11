"""
TTP (Topical and Toxic Prompt) Evaluator.

Uses GPT-4 Omni to classify web pages according to the three-dimensional taxonomy.
"""

import re
import json
import ast
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time

from openai import OpenAI  # type: ignore

from ..utils.taxonomy import HarmLabel

logger = logging.getLogger(__name__)


@dataclass
class TTPResult:
    """Result from TTP evaluation."""
    url: str
    body: str
    predicted_label: HarmLabel
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def is_toxic(self) -> bool:
        return self.predicted_label.is_toxic()

    def is_topical(self) -> bool:
        return self.predicted_label.is_topical()

    def is_safe(self) -> bool:
        return self.predicted_label.is_safe()


class TTPEvaluator:
    """
    Evaluator using TTP prompt with GPT-4.

    Example:
        evaluator = TTPEvaluator(api_key="sk-...")
        result = evaluator.evaluate("https://example.com", "Web page text...")
        print(result.predicted_label.to_dict())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        prompt_path: str = "prompts/TTP/TTP.txt",
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cost_per_token: Optional[float] = None
    ):
        """
        Initialize TTP evaluator.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o)
            prompt_path: Path to TTP prompt file
            temperature: Sampling temperature
            max_retries: Max retry attempts on failure
            retry_delay: Delay between retries in seconds
            cost_per_token: Cost per token in USD for cost estimation (default: uses GPT_4O_COST_PER_TOKEN env var or 0.000008)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Load prompt template
        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"TTP prompt not found: {prompt_path}")

        self.prompt_template = prompt_file.read_text(encoding='utf-8')

        # Extract system message and prepare template
        self._parse_prompt_template()

        # Tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.failed_requests = 0

        # Cost configuration
        # Default GPT-4o pricing: ~$5 per 1M input tokens, ~$15 per 1M output tokens
        # Rough estimate assuming 70/30 input/output split (as of January 2024)
        default_cost_per_token = 0.000008
        self.cost_per_token = cost_per_token if cost_per_token is not None else float(os.getenv("GPT_4O_COST_PER_TOKEN", default_cost_per_token))
        if self.cost_per_token <= 0:
            raise ValueError("cost_per_token must be a positive number")

    def _parse_prompt_template(self):
        """Parse ChatML format prompt into system/user messages."""
        # Extract system message
        system_match = re.search(
            r'<\|im_start\|>system\s*(.*?)<\|im_end\|>',
            self.prompt_template,
            re.DOTALL
        )
        if not system_match:
            raise ValueError("Could not find system message in prompt")

        self.system_message = system_match.group(1).strip()

        # Extract user message template from prompt, fallback to hardcoded if not found
        user_match = re.search(
            r'<\|im_start\|>user\s*(.*?)<\|im_end\|>',
            self.prompt_template,
            re.DOTALL
        )
        if user_match:
            self.user_template = user_match.group(1).strip()
        else:
            # Fallback to hardcoded template
            logger.warning(
                "Prompt template lacked '<|im_start|>user...<|im_end|>' section, "
                "using hardcoded fallback. Template preview: %s...",
                self.prompt_template[:200].replace('\n', '\\n')
            )
            self.user_template = (
                "Web Page Link - #URL#\n"
                "Web Page Content -\n#Body#"
            )

    def evaluate(self, url: str, body: str) -> TTPResult:
        """
        Evaluate a single web page.

        Args:
            url: Web page URL
            body: Web page text content

        Returns:
            TTPResult with predicted labels
        """
        user_message = self.user_template.replace("#URL#", url).replace("#Body#", body)
        last_content: Optional[str] = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature
                )

                self.total_requests += 1

                # Safely extract token usage
                if response.usage is not None and hasattr(response.usage, 'total_tokens'):
                    self.total_tokens += response.usage.total_tokens

                # Safely extract content
                content = None
                if (response.choices and len(response.choices) > 0 and
                    response.choices[0].message and
                    hasattr(response.choices[0].message, 'content')):
                    content = response.choices[0].message.content

                last_content = content

                # Parse response
                label, reasoning = self._parse_response(content)

                return TTPResult(
                    url=url,
                    body=body,
                    predicted_label=label,
                    reasoning=reasoning,
                    raw_response=content
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.failed_requests += 1
                    # Fail-open: return all-safe label, but surface the error + raw response.
                    # This avoids turning evaluation into "everything is toxic" when the model
                    # returns an unparsable response format.
                    fail_open_label = HarmLabel()
                    return TTPResult(
                        url=url,
                        body=body,
                        predicted_label=fail_open_label,
                        raw_response=last_content,
                        error=str(e)
                    )
        
        # This should never be reached, but mypy requires it
        raise RuntimeError("Unexpected execution path in evaluate method")

    def _parse_response(self, content: str) -> tuple[HarmLabel, Optional[str]]:
        """
        Parse GPT response to extract labels and reasoning.

        Expected format:
        <Reasoning>...</Reasoning>
        <Label>{H: Intent-i, IH: None, SE: None, IL: None, SI: None}</Label>
        """
        # Extract reasoning
        reasoning_match = re.search(r'<Reasoning>(.*?)</Reasoning>', content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None

        # Extract label
        label_match = re.search(r'<Label>\s*({.*?})\s*</Label>', content, re.DOTALL)
        label_str: Optional[str] = None
        if label_match:
            label_str = label_match.group(1)
        else:
            # Be permissive: sometimes the model omits the <Label> wrapper but still outputs
            # a dict-like structure we can parse.
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

        # Parse label JSON - handle variations
        try:
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, dict):
                label_dict = parsed
            elif isinstance(parsed, str):
                label_dict = json.loads(parsed)
            else:
                raise ValueError(f"Expected dict or string from literal_eval, got {type(parsed)}")
        except (ValueError, SyntaxError):
            try:
                label_dict = json.loads(label_str)
            except json.JSONDecodeError:
                # Try to parse manually
                label_dict = {}
                for key in ["H", "IH", "SE", "IL", "SI"]:
                    match = re.search(rf'{key}\s*:\s*([^,}}]+)', label_str)
                    if match:
                        value = match.group(1).strip().strip('"\'')
                        label_dict[key] = value

        # Convert to HarmLabel - extract base dimension from "Intent-i" format
        clean_dict = {}
        for key, value in label_dict.items():
            # Extract base dimension (e.g., "Intent-i" -> "intent")
            if isinstance(value, str):
                base_value = value.split('-')[0].lower()
                clean_dict[key] = base_value
            else:
                clean_dict[key] = value

        return HarmLabel.from_dict(clean_dict), reasoning

    def evaluate_batch(
        self,
        samples: List[tuple[str, str]],
        show_progress: bool = True
    ) -> List[TTPResult]:
        """
        Evaluate multiple web pages.

        Args:
            samples: List of (url, body) tuples
            show_progress: Show progress bar

        Returns:
            List of TTPResult objects
        """
        results = []

        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore
                iterator = tqdm(samples, desc="Evaluating")
            except ImportError:
                iterator = samples
        else:
            iterator = samples

        for url, body in iterator:
            result = self.evaluate(url, body)
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self._estimate_cost()
        }

    def _estimate_cost(self) -> float:
        """
        Estimate cost based on token usage.

        Uses configurable cost_per_token value (default: $0.000008 per token).
        Default based on GPT-4o pricing: ~$5 per 1M input tokens, ~$15 per 1M output tokens,
        assuming 70/30 input/output split.

        Returns:
            self.total_tokens * self.cost_per_token
        """
        return self.total_tokens * self.cost_per_token
