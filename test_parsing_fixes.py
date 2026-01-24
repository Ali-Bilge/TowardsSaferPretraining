#!/usr/bin/env python3
"""
Test script to verify the TTP parsing fixes work correctly.
"""

import re
import ast
import json
from typing import Dict, Any, Optional

# Copy the parsing logic from ttp_local.py to test it directly
def test_parse_response(content: str) -> Dict[str, Any]:
    """Test version of the improved parsing logic."""
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
    return clean

def test_parsing_logic():
    """Test the parsing logic directly with mock responses."""
    print("Testing TTP parsing logic...")

    # Mock different types of model outputs
    test_cases = [
        # Case 1: Proper format (should work)
        ('<Label>{H: None, IH: None, SE: Intent-i, IL: None, SI: None}</Label>', "Proper format"),
        # Case 2: Natural language response (DeepSeek style)
        ("Okay, so I'm trying to figure out how to label this web page content based on the guidelines provided. The content is about weight loss pills. H: None, IH: None, SE: Intent-i, IL: None, SI: None", "DeepSeek natural language"),
        # Case 3: JSON format
        ('{H: "None", IH: "None", SE: "Intent-i", IL: "None", SI: "None"}', "JSON format"),
        # Case 4: Mixed natural language
        ("Based on my analysis: H: None, IH: Topical-i, SE: None, IL: None, SI: Intent-i", "Mixed natural language"),
        # Case 5: Empty response (Gemma issue)
        ("", "Empty response"),
        # Case 6: Partial natural language
        ("H: Intent-i, IH: None, SE: None, IL: None, SI: Topical-i", "Key-value pairs only"),
        # Case 7: With reasoning before labels
        ("Let me analyze this content. It seems to contain harmful material. H: Intent-i, IH: None, SE: None, IL: None, SI: None", "Reasoning + labels"),
    ]

    for i, (test_output, description) in enumerate(test_cases, 1):
        print(f"\nTest case {i} ({description}):")
        print(f"Input: {test_output[:80]}{'...' if len(test_output) > 80 else ''}")
        try:
            result = test_parse_response(test_output)
            print(f"  ✓ Parsed successfully: {result}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

def test_edge_cases():
    """Test edge cases that might break parsing."""
    print("\n" + "="*50)
    print("Testing edge cases...")

    edge_cases = [
        # Malformed JSON
        ('{H: None, IH: Intent-i, SE: None, IL: None, SI: "Topical-i"}', "Mixed quotes"),
        # Extra content after labels
        ('H: None, IH: Intent-i, SE: None, IL: None, SI: Topical-i. This concludes my analysis.', "Extra content after"),
        # Case variations
        ('h: none, ih: intent, se: none, il: none, si: topical', "Lowercase keys"),
        # Incomplete
        ('H: None, IH: Intent-i, SE: None', "Only 3 categories"),
        # Wrong format
        ('The content is safe across all categories.', "No labels at all"),
    ]

    for test_output, description in edge_cases:
        print(f"\nEdge case ({description}):")
        print(f"Input: {test_output}")
        try:
            result = test_parse_response(test_output)
            print(f"  ✓ Parsed: {result}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

if __name__ == "__main__":
    test_parsing_logic()
    test_edge_cases()
    print("\nParsing tests completed!")