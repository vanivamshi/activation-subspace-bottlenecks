#!/usr/bin/env python3
"""
Generate actual input-output examples for long context and minimal steering cases.
"""

import torch
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from mamba_model_loader import load_mamba_model_and_tokenizer
from targeted_approach_4 import MemoryEnhancedSteering
from targeted_approach_6 import ComplexReasoningEvaluator

def generate_examples():
    """Generate examples with and without steering"""
    
    logger.info("="*80)
    logger.info("GENERATING INPUT-OUTPUT EXAMPLES")
    logger.info("="*80)
    
    # Load model
    logger.info("\n📦 Loading Mamba-130M...")
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name="state-spaces/mamba-130m-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Model loaded on {device}")
    
    # ============================================================
    # PART 1: Long Context Examples (from targeted_approach_6.py)
    # ============================================================
    logger.info("\n" + "="*80)
    logger.info("PART 1: LONG CONTEXT EXAMPLES (from targeted_approach_6.py)")
    logger.info("="*80)
    
    evaluator = ComplexReasoningEvaluator(tokenizer, device)
    test_suite = evaluator.get_progressive_test_suite()
    long_context_cases = test_suite['level4_long_context'][:2]  # First 2 examples
    
    logger.info("\n📝 Example 1: Alice's favorite color")
    logger.info("-" * 80)
    
    case1 = long_context_cases[0]
    prompt1 = case1['prompt']
    expected1 = case1['expected']
    
    logger.info(f"Input:\n{prompt1}")
    logger.info(f"\nExpected Output: {expected1}")
    
    # Without steering
    inputs = tokenizer(prompt1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_no_steering = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n❌ WITHOUT Steering:")
    logger.info(f"   Output: '{response_no_steering}'")
    
    # With steering (using SimpleSteering from targeted_approach_6.py)
    from targeted_approach_6 import SimpleSteering
    steering = SimpleSteering(model)
    steering.apply_steering(strength=5.0, layer_idx=20)
    
    inputs = tokenizer(prompt1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_with_steering = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n✅ WITH Steering (Cluster 9, Layer 20, 5.0x):")
    logger.info(f"   Output: '{response_with_steering}'")
    
    steering.remove_steering()
    
    logger.info("\n📝 Example 2: Carol's study subject")
    logger.info("-" * 80)
    
    case2 = long_context_cases[1]
    prompt2 = case2['prompt']
    expected2 = case2['expected']
    
    logger.info(f"Input:\n{prompt2}")
    logger.info(f"\nExpected Output: {expected2}")
    
    # Without steering
    inputs = tokenizer(prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_no_steering2 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n❌ WITHOUT Steering:")
    logger.info(f"   Output: '{response_no_steering2}'")
    
    # With steering
    steering = SimpleSteering(model)
    steering.apply_steering(strength=5.0, layer_idx=20)
    
    inputs = tokenizer(prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_with_steering2 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n✅ WITH Steering (Cluster 9, Layer 20, 5.0x):")
    logger.info(f"   Output: '{response_with_steering2}'")
    
    steering.remove_steering()
    
    # ============================================================
    # PART 2: Minimal Steering Examples (from targeted_approach_4.py)
    # ============================================================
    logger.info("\n" + "="*80)
    logger.info("PART 2: MINIMAL STEERING EXAMPLES (from targeted_approach_4.py)")
    logger.info("="*80)
    
    from targeted_approach_2 import EnhancedRealisticEvaluation
    evaluator2 = EnhancedRealisticEvaluation(tokenizer, device)
    test_cases = evaluator2.get_enhanced_test_cases()
    
    # Long context recall examples
    logger.info("\n📝 Long Context Recall - Example 1: Name recall")
    logger.info("-" * 80)
    
    lc_case1 = test_cases['long_context_recall'][0]
    prompt_lc1 = lc_case1['prompt']
    expected_lc1 = lc_case1['expected']
    
    logger.info(f"Input: '{prompt_lc1}'")
    logger.info(f"Expected Output: '{expected_lc1}'")
    
    # Without steering
    inputs = tokenizer(prompt_lc1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_no_steering_lc1 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n❌ WITHOUT Steering:")
    logger.info(f"   Output: '{response_no_steering_lc1}'")
    
    # With minimal steering
    minimal_steering = MemoryEnhancedSteering(model)
    minimal_steering.apply_minimal_interference_steering(layer_idx=20, strength=4.0)
    
    inputs = tokenizer(prompt_lc1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_with_steering_lc1 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n✅ WITH Minimal Steering (Cluster 9, Layer 20, 4.0x):")
    logger.info(f"   Output: '{response_with_steering_lc1}'")
    
    minimal_steering.remove_steering()
    
    logger.info("\n📝 Long Context Recall - Example 2: Code recall")
    logger.info("-" * 80)
    
    lc_case2 = test_cases['long_context_recall'][1]
    prompt_lc2 = lc_case2['prompt']
    expected_lc2 = lc_case2['expected']
    
    logger.info(f"Input: '{prompt_lc2}'")
    logger.info(f"Expected Output: '{expected_lc2}'")
    
    # Without steering
    inputs = tokenizer(prompt_lc2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_no_steering_lc2 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n❌ WITHOUT Steering:")
    logger.info(f"   Output: '{response_no_steering_lc2}'")
    
    # With minimal steering
    minimal_steering = MemoryEnhancedSteering(model)
    minimal_steering.apply_minimal_interference_steering(layer_idx=20, strength=4.0)
    
    inputs = tokenizer(prompt_lc2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_with_steering_lc2 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n✅ WITH Minimal Steering (Cluster 9, Layer 20, 4.0x):")
    logger.info(f"   Output: '{response_with_steering_lc2}'")
    
    minimal_steering.remove_steering()
    
    # Associative recall examples
    logger.info("\n📝 Associative Recall - Example 1: Key-value retrieval")
    logger.info("-" * 80)
    
    ar_case1 = test_cases['associative_recall'][0]
    prompt_ar1 = ar_case1['prompt']
    expected_ar1 = ar_case1['expected']
    
    logger.info(f"Input: '{prompt_ar1}'")
    logger.info(f"Expected Output: '{expected_ar1}'")
    
    # Without steering
    inputs = tokenizer(prompt_ar1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_no_steering_ar1 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n❌ WITHOUT Steering:")
    logger.info(f"   Output: '{response_no_steering_ar1}'")
    
    # With minimal steering
    minimal_steering = MemoryEnhancedSteering(model)
    minimal_steering.apply_minimal_interference_steering(layer_idx=20, strength=4.0)
    
    inputs = tokenizer(prompt_ar1, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_with_steering_ar1 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n✅ WITH Minimal Steering (Cluster 9, Layer 20, 4.0x):")
    logger.info(f"   Output: '{response_with_steering_ar1}'")
    
    minimal_steering.remove_steering()
    
    logger.info("\n📝 Associative Recall - Example 2: Capital city")
    logger.info("-" * 80)
    
    ar_case2 = test_cases['associative_recall'][1]
    prompt_ar2 = ar_case2['prompt']
    expected_ar2 = ar_case2['expected']
    
    logger.info(f"Input: '{prompt_ar2}'")
    logger.info(f"Expected Output: '{expected_ar2}'")
    
    # Without steering
    inputs = tokenizer(prompt_ar2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_no_steering_ar2 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n❌ WITHOUT Steering:")
    logger.info(f"   Output: '{response_no_steering_ar2}'")
    
    # With minimal steering
    minimal_steering = MemoryEnhancedSteering(model)
    minimal_steering.apply_minimal_interference_steering(layer_idx=20, strength=4.0)
    
    inputs = tokenizer(prompt_ar2, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_len = inputs['input_ids'].shape[1]
    response_with_steering_ar2 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    
    logger.info(f"\n✅ WITH Minimal Steering (Cluster 9, Layer 20, 4.0x):")
    logger.info(f"   Output: '{response_with_steering_ar2}'")
    
    minimal_steering.remove_steering()
    
    logger.info("\n" + "="*80)
    logger.info("✅ Examples generated successfully!")
    logger.info("="*80)

if __name__ == "__main__":
    generate_examples()

