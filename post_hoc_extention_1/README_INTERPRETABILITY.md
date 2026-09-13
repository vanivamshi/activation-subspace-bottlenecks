# Interpretability Analysis Workflow

## Overview

The interpretability analysis has been separated into a standalone script to allow:
1. Running interpretability once and reusing results
2. Faster iteration when testing steering strategies
3. Sharing interpretability results across different experiments

## Usage

### Step 1: Run Interpretability Analysis

```bash
python run_interpretability.py
```

This will:
- Load each Mamba variant model
- Run interpretability analysis (bottleneck detection, critical layer identification, cluster neuron discovery)
- Save results to `experiment_logs/interpretability_results.json`

### Step 2: Run Steering Tests

```bash
python steering_s6_types_jamba.py
```

This will:
- Automatically load interpretability results from `experiment_logs/interpretability_results.json` if available
- Skip interpretability analysis if results are found
- Fall back to running interpretability if file doesn't exist or variant is missing

## Output Format

The JSON file contains results for each variant:

```json
{
  "moe-mamba": {
    "variant": "moe-mamba",
    "num_layers": 6,
    "bottleneck_layer": 4,
    "bottleneck_pct": 66.67,
    "critical_layer": 4,
    "critical_layer_pct": 66.67,
    "cluster_neurons": [51, 247, 418, ...],
    "cluster_9_overlap": 1,
    "bottleneck_analysis": {...}
  }
}
```

## Benefits

1. **Speed**: Interpretability analysis only runs once per variant
2. **Reproducibility**: Same interpretability results used across multiple steering experiments
3. **Flexibility**: Can manually edit JSON to test different configurations
4. **Debugging**: Can inspect interpretability results without re-running analysis

