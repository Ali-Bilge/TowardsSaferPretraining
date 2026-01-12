"""
LLM text generation utilities for HAVOC evaluation.

Supports HuggingFace Transformers backend for HPC cluster use.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TransformersGenerator:
    """Text generator using HuggingFace Transformers (local inference)."""

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
        trust_remote_code: bool = False,
    ):
        try:
            import torch  # type: ignore
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

            self.torch = torch
        except ImportError:
            raise RuntimeError("Install: pip install torch transformers accelerate")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if temperature > 0.0 and not do_sample:
            self.do_sample = True
        else:
            self.do_sample = do_sample
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.trust_remote_code = trust_remote_code
        self.device: str

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

        tokenizer_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code, "low_cpu_mem_usage": low_cpu_mem_usage}

        if torch_dtype_param is not None:
            model_kwargs["torch_dtype"] = torch_dtype_param
        if device_map:
            model_kwargs["device_map"] = device_map

        logger.info("Loading %s...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        if self.do_sample and self.temperature <= 0:
            raise ValueError("When do_sample=True, temperature must be > 0")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            logger.debug(
                "Using eos_token as pad_token fallback for %s: pad_token=%s eos_token=%s padding_side=left",
                self.tokenizer.__class__.__name__,
                self.tokenizer.pad_token,
                self.tokenizer.eos_token,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if device_map is not None and device_map != "auto":
            self.device = device_map
        elif device_map == "auto":
            self.device = "auto"
        else:
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            self.device = device
            self.model.to(self.device)  # type: ignore

        self.model.eval()
        logger.info("Model loaded on %s", self.device)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device != "auto":
            inputs = inputs.to(self.device)

        with self.torch.no_grad():
            params = {**inputs, "max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
            if self.do_sample:
                params["temperature"] = self.temperature
            outputs = self.model.generate(**params)

        generated_ids = outputs[0][len(inputs.input_ids[0]) :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    def cleanup(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "torch") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def create_generator(backend: str, model_name: str, **kwargs):
    """Factory to create generators."""
    if backend == "transformers":
        if "max_tokens" in kwargs and "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        return TransformersGenerator(model_name, **kwargs)
    raise ValueError("Unknown backend. Use 'transformers' for HPC clusters.")

