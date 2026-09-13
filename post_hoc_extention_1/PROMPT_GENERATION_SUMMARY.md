# Prompt Generation Summary

## Overview
Updated all steering test files to use 100 prompts per level, optimized for Mamba models (state-space models).

## Changes Made

### 1. Created `prompt_generator_100.py`
- Generates 100 prompts for each of 6 difficulty levels
- Prompts are optimized for Mamba models (simple, direct, sequential)
- Each prompt includes:
  - `prompt`: The test prompt
  - `expected`: Expected answer
  - `alternatives`: Alternative acceptable answers
  - `is_original`: Boolean flag indicating if prompt is original (True) or newly generated (False)

### 2. Updated Steering Files
All steering files now import and use the prompt generator:

- `steering_s6_types_densemamba.py`
- `steering_s6_types_hyena.py`
- `steering_s6_types_mamba2.py`
- `steering_s6_types_miniplm.py`
- `targeted_approach_7.py`

### 3. Prompt Distribution

#### Level 1: Simple Recall (100 prompts)
- **Original prompts**: 3 (marked with `is_original: True`)
- **New prompts**: 97 (marked with `is_original: False`)
- Patterns: Name recall, code recall, arithmetic, colors, items, numbers, ages, cities, animals

#### Level 2: Two-Hop Reasoning (100 prompts)
- **Original prompts**: 4
- **New prompts**: 96
- Patterns: Transitive comparison, conditional logic, indirect reference, addition, age comparison, location chains, ownership transfer, multiplication

#### Level 3: Three-Hop Reasoning (100 prompts)
- **Original prompts**: 3
- **New prompts**: 97
- Patterns: Multi-step height comparison, syllogistic reasoning, spatial reasoning chains, age chains, color inheritance, multi-step arithmetic, ownership chains, size comparison, time sequences, category membership

#### Level 4: Long Context (100 prompts)
- **Original prompts**: 3
- **New prompts**: 97
- Patterns: Person attributes, study subjects, list positions, age recall, city recall, occupation recall, number sequences, color lists, animal preferences, language recall

#### Level 5: Combined Reasoning + Memory (100 prompts)
- **Original prompts**: 3
- **New prompts**: 97
- Patterns: Age comparison + recall, selective arithmetic, multi-step arithmetic reasoning, count items by person, price comparison, age difference calculation, total cost across people, height comparison, average calculation, combined selection and arithmetic

#### Level 6: Stress Test (100 prompts)
- **Original prompts**: 2
- **New prompts**: 98
- Patterns: Person database (10+ facts), garage inventory, student records, product catalog, team roster

## Mamba-Friendly Design Principles

Prompts are designed to be easier for Mamba models by:
1. **Simple, direct questions** - No complex nested structures
2. **Sequential information** - Facts presented in order
3. **Clear patterns** - Repetitive structures that Mamba can learn
4. **Short contexts** - Avoid overly long contexts that challenge state-space models
5. **Direct recall** - Questions that require simple memory retrieval rather than complex reasoning

## Usage

The prompt generator is automatically imported when running any of the steering test files:

```python
from prompt_generator_100 import generate_mamba_friendly_prompts
test_suite = generate_mamba_friendly_prompts()
```

If the import fails, the files fall back to the original test suite (3-4 prompts per level).

## Testing

To run tests with 100 prompts per level:

```bash
# Run for DenseMamba
python steering_s6_types_densemamba.py

# Run for Hyena
python steering_s6_types_hyena.py

# Run for Mamba-2
python steering_s6_types_mamba2.py

# Run for MiniPLM
python steering_s6_types_miniplm.py

# Run targeted approach
python targeted_approach_7.py
```

## Notes

- Original prompts are preserved and marked with `is_original: True`
- New prompts are marked with `is_original: False`
- All prompts follow the same format for consistency
- Prompts are generated deterministically (same seed produces same prompts)
- Each level has exactly 100 prompts total

