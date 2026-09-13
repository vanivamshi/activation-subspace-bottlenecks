# Training on The Pile and Evaluation on Query Datasets

This guide explains how to:
1. Train the Mamba model on The Pile dataset
2. Evaluate the trained model on query datasets (SQuAD, Natural Questions, etc.)
3. Test steering methods on the trained model

## Workflow Overview

```
1. Train Model on The Pile → 2. Test on Query Datasets → 3. Apply Steering
```

## Step 1: Train Model on The Pile

**First, install required dependencies:**
```bash
pip install zstandard
# or if using uv:
# uv pip install zstandard
```

Train the Mamba-130M model on The Pile dataset:

```bash
python train_on_pile.py \
    --model_name state-spaces/mamba-130m-hf \
    --output_dir ./models/mamba_trained_on_pile \
    --num_samples 10000 \
    --epochs 1 \
    --batch_size 4 \
    --lr 5e-5
```

**Parameters:**
- `--model_name`: Starting model (default: `state-spaces/mamba-130m-hf`)
- `--output_dir`: Where to save trained model (default: `./models/mamba_trained_on_pile`)
- `--num_samples`: Number of training samples (default: 10000)
- `--epochs`: Number of training epochs (default: 1)
- `--batch_size`: Training batch size (default: 4)
- `--lr`: Learning rate (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 512)
- `--device`: Device to use (default: `cuda`)

**Note:** Training on The Pile can take time. Start with fewer samples for testing.

## Step 2: Evaluate on Query Datasets

After training, evaluate the model on query datasets using any of the three evaluation scripts:

### Option A: Memory-Focused Evaluation

```bash
python targeted_approach_4.py \
    --trained_model ./models/mamba_trained_on_pile \
    --query_datasets squad natural_questions
```

### Option B: Comprehensive Diagnostic

```bash
python targeted_approach_5.py \
    --trained_model ./models/mamba_trained_on_pile \
    --query_datasets squad triviaqa

python targeted_approach_5.py --query_datasets squad triviaqa - Switched TriviaQA loading to streaming mode (like Natural Questions)
```

### Option C: Capability Assessment

```bash
python targeted_approach_6.py \
    --trained_model ./models/mamba_trained_on_pile \
    --query_datasets squad natural_questions triviaqa
```

**Parameters:**
- `--trained_model`: Path to model trained on The Pile (optional - uses pretrained if not provided)
- `--query_datasets`: Query datasets to test on. Options:
  - `squad`: SQuAD dataset
  - `natural_questions`: Natural Questions dataset
  - `triviaqa`: TriviaQA dataset

## Available Query Datasets

1. **SQuAD** (`squad`): Stanford Question Answering Dataset
   - Reading comprehension questions
   - Context + Question → Answer format

2. **Natural Questions** (`natural_questions`): Google's Natural Questions
   - Real user questions from Google Search
   - Long-form answers

3. **TriviaQA** (`triviaqa`): Trivia Question Answering
   - Trivia questions with evidence
   - Multiple answer formats

## Example Workflow

```bash
# 1. Train on The Pile (start with small sample for testing)
python train_on_pile.py --num_samples 1000 --epochs 1

# 2. Evaluate trained model on SQuAD
python targeted_approach_4.py \
    --trained_model ./models/mamba_trained_on_pile \
    --query_datasets squad

# 3. Compare with pretrained model (no --trained_model flag)
python targeted_approach_4.py --query_datasets squad
```

## Output

Each evaluation script will:
1. Load the trained model (or pretrained if not specified)
2. Load query datasets for testing
3. Run baseline evaluation (no steering)
4. Apply steering methods
5. Compare results
6. Save results to JSON files in `experiment_logs/`

## Notes

- **Training time**: Training on The Pile can be slow. Start with `--num_samples 1000` for testing.
- **Memory**: Large batch sizes may require more GPU memory. Reduce `--batch_size` if needed.
- **Query datasets**: Some datasets may take time to download on first use.
- **Model path**: If `--trained_model` is not provided, scripts use the pretrained model.

## Troubleshooting

**Missing zstandard module:**
```bash
pip install zstandard
# or
uv pip install zstandard
```

**Error loading The Pile:**
- The script tries `monology/pile-uncopyrighted` first (more reliable), then falls back to `EleutherAI/pile`
- Ensure you have internet connection for dataset download
- The Pile dataset requires `trust_remote_code=True` which is now handled automatically
- If you get `ConnectionResetError` or `FileNotFoundError` for `the-eye.eu`:
  - The original Pile hosting may be temporarily unavailable
  - Try again later, or use a different dataset
  - The script will automatically try the alternative dataset first

**Error loading query datasets:**
- Datasets are downloaded from HuggingFace on first use
- Check internet connection and HuggingFace access

**CUDA out of memory:**
- Reduce `--batch_size` in training
- Use `--device cpu` for CPU training (slower)

**Model not found:**
- Ensure training completed successfully
- Check `--output_dir` path matches `--trained_model` path

