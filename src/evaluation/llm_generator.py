"""
LLM text generation utilities for HAVOC evaluation.

Supports HuggingFace Transformers backend for HPC cluster use.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TransformersGenerator:
    """
    Text generator using HuggingFace Transformers (local inference).

    Requires: pip install torch transformers accelerate
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        do_sample: bool = False,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize Transformers generator.

        Args:
            model_name: HuggingFace model name
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            device: Device (cuda/cpu/mps)
            torch_dtype: Data type for model weights (e.g., "float16", "bfloat16")
            device_map: Device mapping strategy ("auto" for multi-GPU)
            low_cpu_mem_usage: Reduce CPU memory during loading
            trust_remote_code: Trust remote code in model files
        """
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            self.torch = torch
        except ImportError:
            raise RuntimeError("Install: pip install torch transformers accelerate")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.trust_remote_code = trust_remote_code

        # Convert torch_dtype string to torch dtype if provided
        torch_dtype_param = None
        if torch_dtype:
            if torch_dtype == "float16":
                torch_dtype_param = torch.float16
            elif torch_dtype == "bfloat16":
                torch_dtype_param = torch.bfloat16
            elif torch_dtype == "float32":
                torch_dtype_param = torch.float32
            else:
                raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

        # Prepare loading kwargs
        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": low_cpu_mem_usage
        }

        if torch_dtype_param:
            model_kwargs["torch_dtype"] = torch_dtype_param

        if device_map:
            model_kwargs["device_map"] = device_map

        logger.info(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        # Validate sampling parameters
        if self.do_sample and self.temperature <= 0:
            raise ValueError("When do_sample=True, temperature must be > 0")

        # Ensure tokenizer has a pad token for batched generation
        # For decoder-only models, using eos_token as pad_token is a common fallback but can affect
        # attention mask handling and generation since padding tokens may be treated as EOS during training
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug(
                f"Using eos_token as pad_token fallback for {self.tokenizer.__class__.__name__}: "
                f"pad_token={self.tokenizer.pad_token}, eos_token={self.tokenizer.eos_token}"
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Handle device placement based on device_map
        if device_map is not None and device_map != "auto":
            # Explicit device_map like "balanced", "sequential", etc. handles placement
            self.device = device_map
            # Don't call .to() as device_map already placed the model
        elif device_map == "auto":
            self.device = "auto"  # Model is distributed across devices
        else:
            # No device_map specified, determine device and place manually
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self.device = device
            self.model.to(self.device)

        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = inputs.to(self.device)

        with self.torch.no_grad():
            params = {
                **inputs,
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample
            }
            if self.do_sample:
                params["temperature"] = self.temperature

            outputs = self.model.generate(**params)

        # Decode only the generated part (not the prompt)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

    def __call__(self, prompt: str) -> str:
        """Make generator callable."""
        return self.generate(prompt)


def create_generator(backend: str, model_name: str, **kwargs):
    """
    Factory function to create generators.

    Args:
        backend: "transformers" (HuggingFace Transformers)
        model_name: HuggingFace model name
        **kwargs: Additional arguments for generator. For transformers backend,
            if max_tokens is passed and max_new_tokens is not, max_tokens will be
            mapped to max_new_tokens for backwards compatibility before constructing
            the TransformersGenerator.

    Returns:
        Generator instance
    """
    if backend == "transformers":
        # Backwards-compatible arg alias: scripts may pass max_tokens, but the
        # generator uses max_new_tokens.
        if "max_tokens" in kwargs and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        return TransformersGenerator(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'transformers' for HPC clusters.")
