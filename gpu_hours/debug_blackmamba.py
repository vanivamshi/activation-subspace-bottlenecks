#!/usr/bin/env python3
"""
Debug script to analyze the 'str' object has no attribute 'contiguous' error
for BlackMamba models.
"""

import torch
import sys
import os
import traceback
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up environment
os.environ.setdefault('TRITON_CACHE_DIR', os.path.expanduser('~/.triton_cache'))
os.makedirs(os.environ['TRITON_CACHE_DIR'], exist_ok=True)

# Check Python headers
python_include_path = os.path.join(sys.prefix, "include", f"python{sys.version_info.major}.{sys.version_info.minor}")
if not os.path.exists(python_include_path):
    conda_base = os.environ.get("CONDA_PREFIX")
    if conda_base:
        conda_python_include = os.path.join(conda_base, "include", f"python{sys.version_info.major}.{sys.version_info.minor}")
        if os.path.exists(conda_python_include):
            python_include_path = conda_python_include
            os.environ['CPATH'] = python_include_path
            os.environ['C_INCLUDE_PATH'] = python_include_path
            print(f"✅ Using conda Python headers from: {python_include_path}")

print("=" * 80)
print("DEBUG SCRIPT: BlackMamba 'str' object has no attribute 'contiguous' Error")
print("=" * 80)
print()

# Step 1: Try to load BlackMamba model
print("Step 1: Loading BlackMamba model...")
try:
    # Try to import from local repo
    repo_path = "/home/vamshi/BlackMamba"
    if os.path.exists(repo_path):
        sys.path.insert(0, repo_path)
        print(f"   ✅ Added {repo_path} to sys.path")
    
    # Try different import paths
    try:
        from mamba_model import MambaModel
        print("   ✅ Successfully imported MambaModel from mamba_model")
    except ImportError:
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaModel
            print("   ✅ Successfully imported MambaModel from mamba_ssm")
        except ImportError:
            raise ImportError("Could not import MambaModel from mamba_model or mamba_ssm")
except Exception as e:
    print(f"   ❌ Failed to import MambaModel: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Load model and tokenizer
print("\nStep 2: Loading model and tokenizer from HuggingFace...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")

try:
    model = MambaModel.from_pretrained("Zyphra/BlackMamba-1.5B")
    model = model.to(device)
    model.eval()
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Zyphra/BlackMamba-1.5B")
    print("   ✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"   ⚠️  Failed to load tokenizer from model, trying GPT2...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("   ✅ Using GPT2 tokenizer as fallback")
    except Exception as e2:
        print(f"   ❌ Failed to load tokenizer: {e2}")
        sys.exit(1)

# Step 3: Test tokenization
print("\nStep 3: Testing tokenization...")
test_text = "Hello, world!"
print(f"   Test text: '{test_text}'")

try:
    # Test 1: Standard tokenization
    print("\n   Test 3.1: Standard tokenizer call...")
    tokenized = tokenizer(test_text, return_tensors="pt")
    print(f"      Type: {type(tokenized)}")
    print(f"      Keys: {tokenized.keys() if isinstance(tokenized, dict) else 'N/A'}")
    
    if isinstance(tokenized, dict):
        input_ids = tokenized.get('input_ids', None)
        print(f"      input_ids type: {type(input_ids)}")
        print(f"      input_ids shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'N/A'}")
        print(f"      input_ids value: {input_ids if isinstance(input_ids, torch.Tensor) else input_ids}")
        
        # Check if input_ids is a string
        if isinstance(input_ids, str):
            print(f"      ⚠️  PROBLEM FOUND: input_ids is a STRING: '{input_ids}'")
            print(f"      This is the source of the error!")
    else:
        # Check if it's a BatchEncoding object
        print(f"      Is BatchEncoding: {hasattr(tokenized, 'input_ids')}")
        if hasattr(tokenized, 'input_ids'):
            input_ids = tokenized.input_ids
            print(f"      input_ids type: {type(input_ids)}")
            print(f"      input_ids shape: {input_ids.shape if isinstance(input_ids, torch.Tensor) else 'N/A'}")
            if isinstance(input_ids, str):
                print(f"      ⚠️  PROBLEM FOUND: input_ids is a STRING: '{input_ids}'")
    
    # Test 2: Check what prepare_tokenizer_inputs would return
    print("\n   Test 3.2: Simulating prepare_tokenizer_inputs...")
    def debug_prepare_tokenizer_inputs(tokenizer, text, device, **kwargs):
        """Debug version of prepare_tokenizer_inputs"""
        print(f"      Input text type: {type(text)}")
        print(f"      Input text value: {repr(text)}")
        
        inputs = tokenizer(text, return_tensors="pt", **kwargs)
        print(f"      Tokenizer output type: {type(inputs)}")
        
        # Handle BatchEncoding objects
        if hasattr(inputs, 'input_ids') or (hasattr(inputs, '__getitem__') and 'input_ids' in inputs):
            print(f"      Detected BatchEncoding-like object")
            if hasattr(inputs, 'input_ids'):
                input_ids = inputs.input_ids
            elif hasattr(inputs, '__getitem__'):
                input_ids = inputs['input_ids']
            else:
                input_ids = None
            
            print(f"      Extracted input_ids type: {type(input_ids)}")
            print(f"      Extracted input_ids value: {repr(input_ids)}")
            
            if input_ids is not None:
                if not isinstance(input_ids, torch.Tensor):
                    print(f"      ⚠️  PROBLEM: input_ids is not a tensor!")
                    if isinstance(input_ids, str):
                        print(f"      ⚠️  CRITICAL: input_ids is a STRING: '{input_ids}'")
                        print(f"      This will cause 'str' object has no attribute 'contiguous' error!")
                    return None
                else:
                    input_ids = input_ids.to(device)
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    return {'input_ids': input_ids}
        
        # Handle dict outputs
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids', None)
            print(f"      Dict input_ids type: {type(input_ids)}")
            if input_ids is not None and isinstance(input_ids, str):
                print(f"      ⚠️  PROBLEM: input_ids in dict is a STRING: '{input_ids}'")
            return inputs
        
        return inputs
    
    processed = debug_prepare_tokenizer_inputs(tokenizer, test_text, device)
    if processed:
        print(f"      Processed output type: {type(processed)}")
        if isinstance(processed, dict):
            input_ids = processed.get('input_ids', None)
            if input_ids is not None:
                print(f"      Final input_ids type: {type(input_ids)}")
                print(f"      Final input_ids is tensor: {isinstance(input_ids, torch.Tensor)}")
    
except Exception as e:
    print(f"   ❌ Error during tokenization test: {e}")
    traceback.print_exc()

# Step 4: Test forward pass
print("\nStep 4: Testing forward pass...")
try:
    # Prepare input
    tokenized = tokenizer(test_text, return_tensors="pt")
    print(f"   Tokenized type: {type(tokenized)}")
    
    # Extract input_ids
    if isinstance(tokenized, dict):
        input_ids = tokenized.get('input_ids', None)
    elif hasattr(tokenized, 'input_ids'):
        input_ids = tokenized.input_ids
    else:
        input_ids = tokenized
    
    print(f"   Extracted input_ids type: {type(input_ids)}")
    print(f"   Extracted input_ids value: {repr(input_ids)}")
    
    # Check if it's a string
    if isinstance(input_ids, str):
        print(f"   ⚠️  CRITICAL ERROR: input_ids is a STRING!")
        print(f"   This will cause: 'str' object has no attribute 'contiguous'")
        print(f"   String value: '{input_ids}'")
        print(f"   String length: {len(input_ids)}")
        sys.exit(1)
    
    # Ensure it's a tensor
    if not isinstance(input_ids, torch.Tensor):
        print(f"   ⚠️  input_ids is not a tensor, converting...")
        if isinstance(input_ids, (list, tuple)):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        else:
            print(f"   ❌ Cannot convert {type(input_ids)} to tensor")
            sys.exit(1)
    
    input_ids = input_ids.to(device)
    print(f"   ✅ input_ids is a tensor: {input_ids.shape}, dtype={input_ids.dtype}")
    
    # Try forward pass
    print("   Running forward pass...")
    with torch.no_grad():
        output = model(input_ids)
    print(f"   ✅ Forward pass successful!")
    print(f"   Output type: {type(output)}")
    if isinstance(output, torch.Tensor):
        print(f"   Output shape: {output.shape}")
    elif isinstance(output, tuple):
        print(f"   Output is tuple with {len(output)} elements")
        if len(output) > 0:
            print(f"   First element shape: {output[0].shape if isinstance(output[0], torch.Tensor) else type(output[0])}")
    
except Exception as e:
    print(f"   ❌ Error during forward pass: {e}")
    print(f"   Error type: {type(e).__name__}")
    traceback.print_exc()
    
    # Check if it's the contiguous error
    if "'str' object has no attribute 'contiguous'" in str(e):
        print("\n   🔍 ROOT CAUSE IDENTIFIED:")
        print("   The error occurs because input_ids is a string instead of a tensor.")
        print("   This happens when the tokenizer returns a BatchEncoding object")
        print("   and we access .input_ids, but somehow it's a string.")
        print("\n   Possible causes:")
        print("   1. Tokenizer is returning a BatchEncoding with string input_ids")
        print("   2. prepare_tokenizer_inputs is not handling BatchEncoding correctly")
        print("   3. The tokenizer itself has a bug")

# Step 5: Test with different input formats
print("\nStep 5: Testing different input formats...")
test_cases = [
    ("String input", test_text),
    ("List of strings", [test_text]),
    ("Already tokenized dict", {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}),
]

for name, test_input in test_cases:
    print(f"\n   Testing: {name}")
    try:
        if isinstance(test_input, str):
            tokenized = tokenizer(test_input, return_tensors="pt")
        elif isinstance(test_input, list):
            tokenized = tokenizer(test_input, return_tensors="pt", padding=True)
        else:
            tokenized = test_input
        
        print(f"      Tokenized type: {type(tokenized)}")
        
        # Extract input_ids
        if isinstance(tokenized, dict):
            input_ids = tokenized.get('input_ids', None)
        elif hasattr(tokenized, 'input_ids'):
            input_ids = tokenized.input_ids
        else:
            input_ids = tokenized
        
        print(f"      input_ids type: {type(input_ids)}")
        
        if isinstance(input_ids, str):
            print(f"      ⚠️  PROBLEM: input_ids is a STRING!")
            continue
        
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device)
            with torch.no_grad():
                _ = model(input_ids)
            print(f"      ✅ Forward pass successful")
        else:
            print(f"      ⚠️  input_ids is not a tensor: {type(input_ids)}")
    
    except Exception as e:
        print(f"      ❌ Error: {e}")
        if "'str' object has no attribute 'contiguous'" in str(e):
            print(f"      🔍 This is the error we're debugging!")

print("\n" + "=" * 80)
print("DEBUG SCRIPT COMPLETE")
print("=" * 80)
print("\nSummary:")
print("If you see 'input_ids is a STRING' anywhere above, that's the root cause.")
print("The fix should ensure input_ids is always converted to a tensor before")
print("being passed to model.forward().")

