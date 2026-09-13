#!/usr/bin/env python3
"""
Debug script to check if all test suite categories are loaded and would be processed.
"""

import sys
import json
from pathlib import Path

# Add the same imports as the main script
try:
    from prompt_generator_100 import generate_mamba_friendly_prompts
    print("✅ Can import prompt_generator_100")
except ImportError as e:
    print(f"❌ Cannot import prompt_generator_100: {e}")

try:
    from dataset_classifier import DatasetQuestionClassifier
    print("✅ Can import dataset_classifier")
except ImportError as e:
    print(f"❌ Cannot import dataset_classifier: {e}")

print("\n" + "="*80)
print("DEBUG: Testing Test Suite Loading")
print("="*80)

# Initialize test suite (same as in steering_s6_types_miniplm.py)
test_suite = {
    'level1_simple_recall': [],
    'level2_two_hop': [],
    'level3_three_hop': [],
    'level4_long_context': [],
    'level5_combined': [],
    'level6_stress_test': [],
}

print(f"\n📋 Initial test_suite keys: {list(test_suite.keys())}")
print(f"📋 Initial test_suite sizes: {[(k, len(v)) for k, v in test_suite.items()]}")

# Load custom prompts
print("\n" + "-"*80)
print("Loading custom prompts...")
try:
    custom_suite = generate_mamba_friendly_prompts()
    print(f"✅ Loaded custom prompts")
    
    custom_mapping = {
        'level1_simple_recall': 'level1_simple_recall',
        'level2_two_hop': 'level2_two_hop',
        'level3_three_hop': 'level3_three_hop',
        'level4_long_context': 'level4_long_context',
        'level5_combined': 'level5_combined',
        'level6_stress_test': 'level6_stress_test',
    }
    
    for custom_key, test_key in custom_mapping.items():
        if custom_key in custom_suite:
            test_suite[test_key].extend(custom_suite[custom_key])
            print(f"   {test_key}: {len(custom_suite[custom_key])} custom prompts")
        else:
            print(f"   ⚠️  {custom_key} not found in custom_suite")
except Exception as e:
    print(f"❌ Error loading custom prompts: {e}")
    import traceback
    traceback.print_exc()

print(f"\n📋 After custom prompts - test_suite sizes: {[(k, len(v)) for k, v in test_suite.items()]}")

# Load dataset prompts
print("\n" + "-"*80)
print("Loading dataset prompts...")
try:
    from dataset_classifier import DatasetQuestionClassifier
    classifier = DatasetQuestionClassifier()
    dataset_file = "experiment_logs/classified_dataset_questions.json"
    
    dataset_questions = classifier.load_classified_questions(dataset_file)
    print(f"✅ Loaded dataset questions from {dataset_file}")
    print(f"   Dataset keys: {list(dataset_questions.keys())}")
    
    category_mapping = {
        'simple_recall': 'level1_simple_recall',
        'two_hop_reasoning': 'level2_two_hop',
        'three_hop_reasoning': 'level3_three_hop',
        'long_context': 'level4_long_context',
        'combined_reasoning_memory': 'level5_combined',
        'stress_test': 'level6_stress_test',
    }
    
    max_per_category = 100
    for dataset_cat, test_cat in category_mapping.items():
        dataset_items = dataset_questions.get(dataset_cat, [])
        print(f"   Dataset category '{dataset_cat}': {len(dataset_items)} items available")
        limited_items = dataset_items[:max_per_category]
        test_suite[test_cat].extend(limited_items)
        if limited_items:
            print(f"   ✅ Added {len(limited_items)} items to {test_cat} (total now: {len(test_suite[test_cat])})")
        else:
            print(f"   ⚠️  No items added to {test_cat}")
except Exception as e:
    print(f"❌ Error loading dataset prompts: {e}")
    import traceback
    traceback.print_exc()

print(f"\n📋 Final test_suite sizes: {[(k, len(v)) for k, v in test_suite.items()]}")

# Check if all categories have cases
print("\n" + "="*80)
print("Category Status Check")
print("="*80)

level_names = {
    'level1_simple_recall': 'Simple Recall',
    'level2_two_hop': 'Two-Hop Reasoning',
    'level3_three_hop': 'Three-Hop Reasoning',
    'level4_long_context': 'Long Context (5-7 facts)',
    'level5_combined': 'Combined Reasoning + Memory',
    'level6_stress_test': 'Stress Test (10+ facts)',
}

all_have_cases = True
for level_key, cases in test_suite.items():
    level_name = level_names.get(level_key, level_key)
    has_cases = len(cases) > 0
    status = "✅" if has_cases else "❌"
    print(f"{status} {level_name:35s}: {len(cases):4d} cases")
    if not has_cases:
        all_have_cases = False

print("\n" + "="*80)
if all_have_cases:
    print("✅ All categories have cases - baseline test should process all 6 categories")
else:
    print("❌ Some categories are empty - baseline test will skip empty categories")

# Simulate the baseline loop (exact same structure as main script)
print("\n" + "="*80)
print("Simulating Baseline Test Loop (exact structure from main script)")
print("="*80)

baseline_results = {}
processed_count = 0
for level_key, cases in test_suite.items():
    level_name = level_names.get(level_key, level_key)
    baseline_correct = 0
    baseline_processed = 0
    
    print(f"\n🔍 Processing {level_name}: {len(cases)} cases")
    
    # Simulate processing first 5 cases to see if loop continues
    for i, case in enumerate(cases[:5]):  # Just check first 5
        if 'prompt' not in case:
            print(f"   ⚠️  Case {i+1} missing 'prompt' key")
            continue
        if 'expected' not in case:
            print(f"   ⚠️  Case {i+1} missing 'expected' key")
            continue
        baseline_processed += 1
        # Simulate: mark as correct for testing
        baseline_correct += 1
    
    if len(cases) > 0:
        baseline_acc = baseline_correct / baseline_processed if baseline_processed > 0 else 0.0
        baseline_results[level_key] = {
            'correct': baseline_correct,
            'accuracy': baseline_acc,
            'processed': baseline_processed
        }
        print(f"   {level_name:30s}: {baseline_acc*100:5.1f}% ({baseline_correct}/{baseline_processed} processed, {len(cases)} total)")
        processed_count += 1
    else:
        print(f"   ⚠️  SKIPPED - No cases to process")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Categories in test_suite: {len(test_suite)}")
print(f"Categories with cases: {sum(1 for v in test_suite.values() if len(v) > 0)}")
print(f"Categories that would be processed: {len(baseline_results)}")
print(f"\nCategories that would be processed: {list(baseline_results.keys())}")
print(f"Categories that would be skipped: {[k for k in test_suite.keys() if k not in baseline_results]}")
print(f"\n⚠️  If Stress Test is missing, check if loop exits early or if exception is caught")

# Test exact iteration order
print("\n" + "="*80)
print("Testing Exact Iteration (mimicking main script)")
print("="*80)

# Create a copy to test iteration
test_suite_copy = dict(test_suite)  # Explicit copy
iteration_order = []
for level_key, cases in test_suite_copy.items():
    iteration_order.append(level_key)
    level_name = level_names.get(level_key, level_key)
    print(f"  Iteration {len(iteration_order)}: {level_key} -> {level_name} ({len(cases)} cases)")

print(f"\n✅ Iteration order: {iteration_order}")
print(f"✅ Total iterations: {len(iteration_order)}")
if 'level6_stress_test' not in iteration_order:
    print(f"❌ ERROR: level6_stress_test is NOT in iteration order!")
elif iteration_order[-1] != 'level6_stress_test':
    print(f"⚠️  WARNING: level6_stress_test is not last (it's at position {iteration_order.index('level6_stress_test') + 1})")
else:
    print(f"✅ level6_stress_test is last in iteration order (position {len(iteration_order)})")

# Check dictionary iteration order
print("\n" + "="*80)
print("Dictionary Iteration Order Check")
print("="*80)
print("Python dict iteration order (should be insertion order in Python 3.7+):")
for i, (key, value) in enumerate(test_suite.items(), 1):
    print(f"  {i}. {key}: {len(value)} cases")
