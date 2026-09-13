# MambaByte Setup Guide

MambaByte is a token-free selective state space model implementation. This guide explains how to set it up and use it with the steering script.

## What is MambaByte?

MambaByte is a token-free architecture that processes raw bytes instead of tokens. Based on the paper "MambaByte: Token-free Selective State Space Model" and implemented in PyTorch and Zeta.

- **GitHub**: https://github.com/kyegomez/MambaByte
- **Architecture**: Token-free Selective State Space Model
- **Key Feature**: Processes raw bytes directly, no tokenization needed

## Installation

### Method 1: Install from PyPI (Recommended)

```bash
pip install mambabyte
```

### Method 2: Install from GitHub

```bash
git clone https://github.com/kyegomez/MambaByte.git
cd MambaByte
pip install -e .
```

### Method 3: Use Local Repository

If you've cloned the repository locally:

```bash
# Set environment variable
export MAMBABYTE_REPO_PATH=/path/to/MambaByte

# Or update the config in steering_s6_types_granite.py:
# "repo_path": "/path/to/MambaByte"
```

## Configuration

MambaByte is already configured in `MAMBA_VARIANTS`. You can customize the model configuration:

```python
MAMBA_VARIANTS = {
    "mambabyte": {
        "model_name": None,  # Token-free, no standard tokenizer
        "architecture": "mambabyte",
        "github": "https://github.com/kyegomez/MambaByte",
        "use_custom_loader": True,
        "repo_path": "/home/vamshi/MambaByte",  # Optional: local repo path
        "model_config": {
            "dim": 512,          # Model dimension
            "depth": 6,          # Number of layers
            "dt_rank": 16,       # Time step rank
            "d_state": 16,       # State dimension
            "expand_factor": 2,  # Expansion factor
            "d_conv": 4,         # Convolution dimension
            "dt_min": 0.001,     # Min time step
            "dt_max": 0.1,       # Max time step
            "pscan": True        # Use parallel scan
        }
    }
}
```

## Usage

### Running the Script

```bash
python steering_s6_types_granite.py
```

The script will automatically detect and load MambaByte if it's configured.

### Programmatic Usage

```python
from steering_s6_types_granite import load_model_variant

# Load MambaByte model
model, tokenizer = load_model_variant("mambabyte", device="cuda")

# Note: The tokenizer is a byte-level wrapper for compatibility
# MambaByte processes raw bytes, not tokens
```

### Direct Usage (if needed)

```python
from mambabyte.model import MambaConfig, Mamba

# Create config
config = MambaConfig(
    dim=512,
    depth=6,
    dt_rank=16,
    d_state=16,
    expand_factor=2,
    d_conv=4
)

# Create model
model = Mamba(config)
```

## Model Architecture

MambaByte uses a token-free architecture:

- **Input**: Raw bytes (0-255)
- **Processing**: Selective State Space Model (SSM)
- **Output**: Byte-level predictions

The implementation includes:
- Parallel scan for efficient computation
- Configurable state dimensions
- Convolutional layers for local patterns

## Tokenizer Note

Since MambaByte is token-free, the script includes a `ByteTokenizer` wrapper that:
- Converts text to bytes for input
- Converts byte outputs back to text
- Maintains compatibility with the rest of the codebase

## Example Configuration

Here's an example of a larger MambaByte model:

```python
"mambabyte-large": {
    "architecture": "mambabyte",
    "use_custom_loader": True,
    "model_config": {
        "dim": 1024,
        "depth": 12,
        "d_state": 32,
        "expand_factor": 2,
        "d_conv": 4,
        "pscan": True
    }
}
```

## Troubleshooting

### Import Error: "No module named 'mambabyte'"

**Solution**: Install MambaByte:
```bash
pip install mambabyte
```

### Repository Not Found

**Solution**: Clone the repository or set the path:
```bash
git clone https://github.com/kyegomez/MambaByte.git /home/vamshi/MambaByte
export MAMBABYTE_REPO_PATH=/home/vamshi/MambaByte
```

### Model Configuration Issues

**Solution**: Check that all required config parameters are set. Default values are provided, but you can customize them in the `model_config` dictionary.

## References

- **GitHub Repository**: https://github.com/kyegomez/MambaByte
- **Paper**: "MambaByte: Token-free Selective State Space Model"
- **Installation**: `pip install mambabyte`

## Notes

- MambaByte is token-free, so it processes raw bytes directly
- The model configuration is flexible and can be adjusted for different use cases
- Parallel scan (pscan) is enabled by default for better performance
- The model works with byte-level data, making it suitable for various input types

