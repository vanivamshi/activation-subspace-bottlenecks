#!/usr/bin/env python3
"""
Debug script to analyze tokenizer output and find where strings might be passed
instead of tensors for BlackMamba models.
"""

import torch
import sys
from transformers import AutoTokenizer

print("=" * 80)
print("DEBUG SCRIPT: Tokenizer Output Analysis")
print("=" * 80)
print()

# Test 1: Load tokenizer
print("Step 1: Loading tokenizer...")
try:
    # Try BlackMamba tokenizer first
    try:
        tokenizer = AutoTokenizer.from_pretrained("Zyphra/BlackMamba-1.5B")
        print("   ✅ Loaded tokenizer from BlackMamba-1.5B")
    except:
        print("   ⚠️  Could not load from BlackMamba, using GPT2...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("   ✅ Loaded GPT2 tokenizer as fallback")
except Exception as e:
    print(f"   ❌ Failed to load tokenizer: {e}")
    sys.exit(1)

# Test 2: Test tokenization
print("\nStep 2: Testing tokenization with different inputs...")
test_texts = [
    "Hello, world!",
    "The quick brown fox",
    "",  # Empty string
]

for test_text in test_texts:
    print(f"\n   Testing: '{test_text}'")
    
    # Test 2.1: Standard tokenization
    print(f"      Test 2.1: Standard tokenizer call...")
    try:
        tokenized = tokenizer(test_text, return_tensors="pt")
        print(f"         Type: {type(tokenized)}")
        print(f"         Is dict: {isinstance(tokenized, dict)}")
        print(f"         Is BatchEncoding: {hasattr(tokenized, 'input_ids')}")
        
        # Extract input_ids
        if isinstance(tokenized, dict):
            input_ids = tokenized.get('input_ids', None)
            print(f"         Dict has 'input_ids': {'input_ids' in tokenized}")
        elif hasattr(tokenized, 'input_ids'):
            input_ids = tokenized.input_ids
            print(f"         BatchEncoding has input_ids attribute")
        else:
            input_ids = None
            print(f"         ⚠️  Cannot extract input_ids")
        
        if input_ids is not None:
            print(f"         input_ids type: {type(input_ids)}")
            print(f"         input_ids is tensor: {isinstance(input_ids, torch.Tensor)}")
            print(f"         input_ids is string: {isinstance(input_ids, str)}")
            print(f"         input_ids is list: {isinstance(input_ids, list)}")
            
            if isinstance(input_ids, str):
                print(f"         ⚠️  PROBLEM FOUND: input_ids is a STRING!")
                print(f"         String value: '{input_ids}'")
                print(f"         This will cause 'str' object has no attribute 'contiguous'")
            elif isinstance(input_ids, torch.Tensor):
                print(f"         ✅ input_ids is a tensor: shape={input_ids.shape}, dtype={input_ids.dtype}")
            else:
                print(f"         ⚠️  input_ids is unexpected type: {type(input_ids)}")
                print(f"         Value: {repr(input_ids)}")
        else:
            print(f"         ⚠️  input_ids is None")
    
    except Exception as e:
        print(f"         ❌ Error: {e}")
        traceback.print_exc()

# Test 3: Simulate prepare_tokenizer_inputs logic
print("\nStep 3: Simulating prepare_tokenizer_inputs logic...")

def simulate_prepare_tokenizer_inputs(tokenizer, text, device, **kwargs):
    """Simulate the prepare_tokenizer_inputs function"""
    print(f"      Input: '{text}'")
    
    inputs = tokenizer(text, return_tensors="pt", **kwargs)
    print(f"      Tokenizer output type: {type(inputs)}")
    
    # Check if it's a BatchEncoding object (dict-like but not isinstance(dict))
    if hasattr(inputs, 'input_ids') or (hasattr(inputs, '__getitem__') and 'input_ids' in inputs):
        print(f"      Detected BatchEncoding-like object")
        
        # Extract input_ids
        if hasattr(inputs, 'input_ids'):
            input_ids = inputs.input_ids
            print(f"      Extracted via .input_ids attribute")
        elif hasattr(inputs, '__getitem__'):
            input_ids = inputs['input_ids']
            print(f"      Extracted via ['input_ids']")
        else:
            input_ids = None
            print(f"      ⚠️  Could not extract input_ids")
        
        if input_ids is not None:
            print(f"      Extracted input_ids type: {type(input_ids)}")
            print(f"      Extracted input_ids value: {repr(input_ids)}")
            
            # Check if it's a string
            if isinstance(input_ids, str):
                print(f"      ⚠️  CRITICAL: input_ids is a STRING!")
                print(f"      This is the root cause of the error!")
                return None
            
            # Convert to tensor if needed
            if not isinstance(input_ids, torch.Tensor):
                if isinstance(input_ids, (list, tuple)):
                    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
                    print(f"      Converted list to tensor")
                else:
                    print(f"      ⚠️  Cannot convert {type(input_ids)} to tensor")
                    return None
            else:
                input_ids = input_ids.to(device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                print(f"      ✅ Final input_ids: shape={input_ids.shape}, dtype={input_ids.dtype}")
            
            return {'input_ids': input_ids}
        else:
            print(f"      ⚠️  input_ids is None")
            return None
    
    # Handle dict outputs
    elif isinstance(inputs, dict):
        print(f"      Detected dict output")
        input_ids = inputs.get('input_ids', None)
        if input_ids is not None:
            print(f"      input_ids type: {type(input_ids)}")
            if isinstance(input_ids, str):
                print(f"      ⚠️  PROBLEM: input_ids in dict is a STRING!")
                return None
            input_ids = input_ids.to(device) if isinstance(input_ids, torch.Tensor) else torch.tensor([input_ids], dtype=torch.long).to(device)
            return {'input_ids': input_ids}
        else:
            print(f"      ⚠️  No 'input_ids' in dict")
            return None
    
    else:
        print(f"      ⚠️  Unknown tokenizer output type: {type(inputs)}")
        return None

for test_text in test_texts[:2]:  # Skip empty string for this test
    print(f"\n   Testing prepare_tokenizer_inputs with: '{test_text}'")
    result = simulate_prepare_tokenizer_inputs(tokenizer, test_text, "cuda" if torch.cuda.is_available() else "cpu")
    if result:
        print(f"      ✅ Result: {type(result)}, input_ids shape: {result['input_ids'].shape}")
    else:
        print(f"      ❌ Failed to prepare inputs")

# Test 4: Check what happens when we access attributes
print("\nStep 4: Testing attribute access patterns...")
test_text = "Hello"
tokenized = tokenizer(test_text, return_tensors="pt")

print(f"   Tokenized object type: {type(tokenized)}")
print(f"   Tokenized object dir: {[x for x in dir(tokenized) if not x.startswith('_')][:10]}")

# Try different ways to access input_ids
access_methods = [
    ("getattr(tokenized, 'input_ids')", lambda: getattr(tokenized, 'input_ids', None)),
    ("tokenized['input_ids']", lambda: tokenized['input_ids'] if hasattr(tokenized, '__getitem__') else None),
    ("tokenized.input_ids", lambda: tokenized.input_ids if hasattr(tokenized, 'input_ids') else None),
]

for method_name, method_func in access_methods:
    try:
        result = method_func()
        print(f"   {method_name}: type={type(result)}, value={repr(result)[:50]}")
        if isinstance(result, str):
            print(f"      ⚠️  PROBLEM: This method returns a STRING!")
    except Exception as e:
        print(f"   {method_name}: Error - {e}")

print("\n" + "=" * 80)
print("DEBUG SCRIPT COMPLETE")
print("=" * 80)
print("\nSummary:")
print("If any input_ids is a STRING above, that's the root cause.")
print("The fix should ensure input_ids is always converted to a tensor.")

