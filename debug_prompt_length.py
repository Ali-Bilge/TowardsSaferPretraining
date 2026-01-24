#!/usr/bin/env python3
"""
Debug script to check prompt lengths and see if truncation is causing issues.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Mock the tokenizer to test prompt formatting without loading the actual model
class MockTokenizer:
    def __init__(self):
        self.model_max_length = 8192

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Simulate what a chat template might produce
        result = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                result += f"System: {content}\n\n"
            elif role == 'user':
                result += f"User: {content}\n\n"
            elif role == 'assistant':
                result += f"Assistant: {content}\n\n"
        if add_generation_prompt:
            result += "Assistant: "
        return result

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        # Mock tokenization
        tokens = len(text.split()) * 1.3  # Rough token estimate
        return {"input_ids": list(range(int(tokens)))}

def test_prompt_formatting():
    """Test prompt formatting and lengths."""
    from src.clients.ttp_local import TransformersTTPClient

    # Create a mock client instance to test formatting
    client = TransformersTTPClient.__new__(TransformersTTPClient)
    client.prompt_template = """<|im_start|>system
You are an expert content moderator...
[full prompt content would be here]
<|im_start|>user
Web Page Link - #URL#
Web Page Content -
#Body#
<|im_end|>
<|im_start|>assistant
"""

    client._parse_prompt_template()

    # Test with a sample
    test_url = "https://example.com/test"
    test_body = "This is a test web page content about weight loss pills." * 50  # Make it longer

    # Test prompt formatting
    client.tokenizer = MockTokenizer()
    client._format_prompt = lambda user_msg: client.tokenizer.apply_chat_template([
        {"role": "system", "content": client.system_message},
        {"role": "user", "content": user_msg}
    ], tokenize=False, add_generation_prompt=True)

    user_message = client.user_template.replace("#URL#", test_url).replace("#Body#", test_body)
    user_message += "\n\n# OUTPUT FORMAT\nYou must respond with ONLY the label in this exact format:\n<Label>{H: None|Topical-i|Intent-i, IH: None|Topical-i|Intent-i, SE: None|Topical-i|Intent-i, IL: None|Topical-i|Intent-i, SI: None|Topical-i|Intent-i}</Label>\nDo not include any other text, reasoning, or explanation."

    formatted_prompt = client._format_prompt(user_message)

    print(f"User message length: {len(user_message)} chars")
    print(f"Formatted prompt length: {len(formatted_prompt)} chars")
    print(f"System message length: {len(client.system_message)} chars")
    print(f"Number of <|im_start|> blocks: {client.prompt_template.count('<|im_start|>')}")

    # Estimate token count
    estimated_tokens = len(formatted_prompt.split()) * 1.3
    print(f"Estimated token count: {int(estimated_tokens)}")

    print("\nFirst 500 chars of formatted prompt:")
    print(repr(formatted_prompt[:500]))

    print("\nLast 500 chars of formatted prompt:")
    print(repr(formatted_prompt[-500:]))

if __name__ == "__main__":
    test_prompt_formatting()