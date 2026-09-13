# Dataset Classification and Evaluation Guide

This guide explains how to use classified dataset questions for evaluation alongside custom prompts.

## Overview

The system now supports:
1. **Custom Prompts**: Synthetic prompts from `prompt_generator_100.py` (your original test suite)
2. **Dataset Prompts**: Questions from real datasets (HotpotQA, TriviaQA, MuSiQue, DROP) classified into your test categories
3. **Combined Evaluation**: Run both custom and dataset prompts together

## Categories

Questions are classified into these categories:
- **Simple Recall**: Direct fact retrieval
- **Two-Hop Reasoning**: Requires 2 steps of reasoning
- **Three-Hop Reasoning**: Requires 3+ steps of reasoning
- **Long Context (5-7 facts)**: Multiple facts in context
- **Combined Reasoning + Memory**: Requires both reasoning and memory
- **Stress Test (10+ facts)**: Very long context with many facts
- **Query Dataset Tasks**: TriviaQA questions

## Step 1: Create Classified Dataset File

First, create a file with classified questions from datasets:

```bash
python targeted_approach_7.py --create_dataset_file \
    --datasets_to_classify squad hotpotqa triviaqa musique drop natural_questions \
    --num_per_dataset 100 \
    --target_per_category 100 \
    --dataset_file experiment_logs/classified_dataset_questions.json
```

**Note**: 
- **SQuAD** is important for Simple Recall, Long Context, and Stress Test categories
- **Natural Questions** is important for Long Context and Stress Test (very long contexts)
- Other datasets focus on reasoning categories

This will:
- Load questions from SQuAD, HotpotQA, TriviaQA, MuSiQue, DROP, and Natural Questions
- Classify each question into one of the categories
- **Automatically ensure ~100 questions per category** (fills gaps by loading more samples)
- Save to `experiment_logs/classified_dataset_questions.json`

## Step 2: Run Evaluation

### Option A: Use Only Custom Prompts (Default)
```bash
python targeted_approach_7.py \
    --trained_model ./models/mamba_trained_on_pile \
    --use_custom_prompts
```

### Option B: Use Only Dataset Prompts
```bash
python targeted_approach_7.py \
    --trained_model ./models/mamba_trained_on_pile \
    --no_custom_prompts \
    --use_dataset_prompts \
    --dataset_file experiment_logs/classified_dataset_questions.json \
    --max_per_category 100
```

### Option C: Use Both Custom and Dataset Prompts
```bash
python targeted_approach_7.py \
    --trained_model ./models/mamba_trained_on_pile \
    --use_custom_prompts \
    --use_dataset_prompts \
    --dataset_file experiment_logs/classified_dataset_questions.json \
    --max_per_category 100
```

## Command-Line Options

- `--use_custom_prompts`: Use custom prompts (default: True)
- `--no_custom_prompts`: Don't use custom prompts
- `--use_dataset_prompts`: Use classified dataset prompts
- `--dataset_file PATH`: Path to classified dataset questions JSON file
- `--max_per_category N`: Maximum questions per category from datasets (default: 100)
- `--create_dataset_file`: Create classified dataset file before evaluation
- `--datasets_to_classify`: Datasets to classify (default: squad hotpotqa triviaqa musique drop natural_questions)
- `--num_per_dataset N`: Initial number of samples per dataset when creating file (default: 100)
- `--target_per_category N`: Target number of questions per category (default: 100)

## Dataset Mapping

- **SQuAD** → Simple Recall, Long Context (5-7 facts), Stress Test (10+ facts)
  - Has varied context lengths and question complexities
  - Good source for all three categories
  
- **Natural Questions** → Long Context, Stress Test (10+ facts)
  - Very long contexts with many facts
  - Excellent for stress testing
  
- **HotpotQA** → Two-Hop Reasoning, Three-Hop Reasoning
  - Designed for multi-hop questions with supporting facts
  
- **TriviaQA** → Query Dataset Tasks
  - Trivia questions for general knowledge testing
  
- **MuSiQue** → Two-Hop Reasoning, Three-Hop Reasoning
  - Has explicit hop counts in metadata
  
- **DROP** → Combined Reasoning + Memory
  - Numeric/discrete reasoning tasks

## Example Workflow

```bash
# 1. Create classified dataset file (one time)
python targeted_approach_7.py --create_dataset_file \
    --datasets_to_classify hotpotqa triviaqa musique drop \
    --num_per_dataset 100

# 2. Run evaluation with both custom and dataset prompts
python targeted_approach_7.py \
    --trained_model ./models/mamba_trained_on_pile \
    --use_custom_prompts \
    --use_dataset_prompts \
    --max_per_category 100

# 3. Compare results - you'll see both custom and dataset questions in results
```

## Results Format

Results will show performance for each category, with questions from:
- Custom prompts (if `--use_custom_prompts`)
- Dataset prompts (if `--use_dataset_prompts`)
- Both combined (if both options are used)

The results table will show:
```
Level                          Tasks    Baseline     With Steering   Change       Status
--------------------------------------------------------------------------------
Simple Recall                  200       95.0%         93.0%           -2.0%        ❌ NEGATIVE
Two-Hop Reasoning              200       45.0%         44.0%           -1.0%        ❌ NEGATIVE
...
```

Where "Tasks" column shows total questions (custom + dataset combined).

## Notes

- The classified dataset file is saved once and can be reused across multiple runs
- Questions are classified automatically based on:
  - Number of supporting facts
  - Context length
  - Question complexity
  - Explicit hop counts (for MuSiQue, HotpotQA)
- TriviaQA questions are always assigned to "Query Dataset Tasks" category
- DROP questions are assigned to "Combined Reasoning + Memory" category
