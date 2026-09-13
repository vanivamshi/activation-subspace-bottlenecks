# How to Clone Granite 4.0 Micro Model Locally

The Granite model files are hosted on **HuggingFace**, not GitHub. The GitHub repository (https://github.com/ibm-granite/granite-4.0-language-models) contains documentation and examples only.

## Method 1: Clone from HuggingFace using git-lfs (RECOMMENDED)

This is the most reliable method for downloading large models:

```bash
# Install git-lfs if not already installed
# Ubuntu/Debian:
sudo apt-get install git-lfs

# macOS:
brew install git-lfs

# Initialize git-lfs
git lfs install

# Clone the model repository
git clone https://huggingface.co/ibm-granite/granite-4.0-micro

# Set environment variable to use the cloned model
export GRANITE_MODEL_PATH=./granite-4.0-micro

# Run your script
python steering_s6_types_granite.py
```

## Method 2: Using huggingface-cli

```bash
# Install huggingface-cli if needed
pip install huggingface_hub[cli]

# Download the model
huggingface-cli download ibm-granite/granite-4.0-micro --local-dir ./granite-4.0-micro

# Set environment variable
export GRANITE_MODEL_PATH=./granite-4.0-micro

# Run your script
python steering_s6_types_granite.py
```

## Method 3: Using Python script

```python
from huggingface_hub import snapshot_download

model_name = "ibm-granite/granite-4.0-micro"
download_dir = "./granite-4.0-micro"

snapshot_download(
    repo_id=model_name,
    local_dir=download_dir,
    local_dir_use_symlinks=False
)

# Then set: export GRANITE_MODEL_PATH=./granite-4.0-micro
```

## Method 4: Update config in code

You can also directly specify the path in the `MAMBA_VARIANTS` config:

```python
MAMBA_VARIANTS = {
    "granite-4.0-micro": {
        "model_name": "ibm-granite/granite-4.0-micro",
        "local_path": "/path/to/granite-4.0-micro",  # Add this line
        "architecture": "granite",
        # ... rest of config
    }
}
```

## Important Notes

1. **Model Size**: The granite-4.0-micro model is approximately **6.8GB** (3B parameters)
2. **Git LFS**: If using `git clone`, make sure `git-lfs` is installed, otherwise you'll only get pointer files
3. **Disk Space**: Ensure you have at least **10GB** free space
4. **Network**: Cloning/downloading may take time depending on your internet connection

## Troubleshooting

### If git clone fails:
- Make sure `git-lfs` is installed and initialized: `git lfs install`
- Check if you have enough disk space
- Try downloading during off-peak hours

### If download is interrupted:
- The script will automatically try to resume from cache
- You can also manually resume by running the same command again

### If model still doesn't load:
- Verify the path: `ls -lh /path/to/granite-4.0-micro`
- Check that all files are present (config.json, tokenizer files, .safetensors files)
- Make sure the path is absolute or relative to where you run the script

## Verification

After cloning/downloading, verify the model files are present:

```bash
ls -lh granite-4.0-micro/
# Should see files like:
# - config.json
# - model.safetensors.index.json
# - model-00001-of-00002.safetensors
# - model-00002-of-00002.safetensors
# - tokenizer.json (or tokenizer files)
```

