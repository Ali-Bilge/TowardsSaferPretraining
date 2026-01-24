#!/usr/bin/env python3
"""
Test script to verify HuggingFace access to Llama Guard 3 8B (gated model).
Run this before submitting cluster jobs to ensure your token works.

Usage:
    # Option 1: Set token as environment variable
    export HF_TOKEN="your_token_here"
    python test_llama_guard_access.py

    # Option 2: Pass token as argument
    python test_llama_guard_access.py --token "your_token_here"

    # Option 3: Login interactively first
    huggingface-cli login
    python test_llama_guard_access.py
"""

import argparse
import os
import sys

def test_llama_guard_access(token=None):
    """Test access to the gated Llama Guard 3 8B model."""

    print("=" * 60)
    print("Llama Guard 3 8B Access Test")
    print("=" * 60)

    # Step 1: Check/set token
    print("\n[1/4] Checking HuggingFace token...")

    if token:
        os.environ["HF_TOKEN"] = token
        print("      Using provided token")
    elif os.environ.get("HF_TOKEN"):
        print("      Using HF_TOKEN environment variable")
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("      Using HUGGING_FACE_HUB_TOKEN environment variable")
    else:
        print("      No token provided, will try cached credentials")

    # Step 2: Test huggingface_hub login
    print("\n[2/4] Testing HuggingFace Hub connection...")
    try:
        from huggingface_hub import HfApi, whoami

        api = HfApi()
        user_info = whoami(token=token)
        print(f"      Logged in as: {user_info.get('name', 'Unknown')}")
        print(f"      Account type: {user_info.get('type', 'Unknown')}")
    except Exception as e:
        print(f"      ERROR: Failed to authenticate with HuggingFace")
        print(f"      {type(e).__name__}: {e}")
        print("\n      Make sure you have:")
        print("      1. Created a HuggingFace account")
        print("      2. Generated an access token at https://huggingface.co/settings/tokens")
        print("      3. Accepted the Llama Guard license at https://huggingface.co/meta-llama/Llama-Guard-3-8B")
        return False

    # Step 3: Test model access (config only - fast)
    print("\n[3/4] Testing access to meta-llama/Llama-Guard-3-8B...")
    model_id = "meta-llama/Llama-Guard-3-8B"

    try:
        from huggingface_hub import model_info

        info = model_info(model_id, token=token)
        print(f"      Model found: {info.modelId}")
        print(f"      Downloads: {info.downloads:,}")
        print(f"      Gated: {info.gated}")
    except Exception as e:
        print(f"      ERROR: Cannot access model")
        print(f"      {type(e).__name__}: {e}")

        if "403" in str(e) or "gated" in str(e).lower():
            print("\n      You need to accept the license agreement:")
            print(f"      https://huggingface.co/{model_id}")
            print("      Click 'Agree and access repository' then try again.")
        return False

    # Step 4: Test loading tokenizer (lightweight test)
    print("\n[4/4] Testing tokenizer download...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True
        )
        print(f"      Tokenizer loaded successfully!")
        print(f"      Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"      ERROR: Failed to load tokenizer")
        print(f"      {type(e).__name__}: {e}")
        return False

    # Success!
    print("\n" + "=" * 60)
    print("SUCCESS! Your HuggingFace token has access to Llama Guard 3.")
    print("You can now run the full evaluation on the cluster.")
    print("=" * 60)

    return True


def test_model_inference(token=None):
    """Optional: Run a quick inference test (requires GPU or patience)."""

    print("\n[Optional] Running minimal inference test...")
    print("          (This will download ~16GB of model weights)")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_id = "meta-llama/Llama-Guard-3-8B"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"          Device: {device}")

        if device == "cpu":
            print("          WARNING: Running on CPU will be slow!")

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )

        # Simple test input
        test_input = "Is this content safe?"
        inputs = tokenizer(test_input, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"          Test output: {response[:100]}...")
        print("          Inference test PASSED!")
        return True

    except Exception as e:
        print(f"          Inference test failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HuggingFace access to Llama Guard 3")
    parser.add_argument("--token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--full-test", action="store_true", help="Also run inference test (downloads full model)")
    args = parser.parse_args()

    success = test_llama_guard_access(args.token)

    if success and args.full_test:
        test_model_inference(args.token)

    sys.exit(0 if success else 1)
