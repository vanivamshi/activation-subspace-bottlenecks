#!/bin/bash
# Script to manually download Granite 4.0 Micro model
# This script provides multiple methods to download the model
# NOTE: Model files are on HuggingFace, not GitHub!
# GitHub repo (https://github.com/ibm-granite/granite-4.0-language-models) has docs only

MODEL_NAME="ibm-granite/granite-4.0-micro"
DOWNLOAD_DIR="${1:-./granite-4.0-micro}"  # Use first argument or default to ./granite-4.0-micro

echo "=========================================="
echo "Granite 4.0 Micro Model Download Script"
echo "=========================================="
echo ""
echo "Model: $MODEL_NAME"
echo "Download directory: $DOWNLOAD_DIR"
echo ""
echo "NOTE: Model files are on HuggingFace, not GitHub!"
echo "GitHub repo contains documentation/examples only."
echo ""

# Method 1: Using git-lfs clone (RECOMMENDED - Most Reliable)
echo "Method 1: Using git-lfs clone (RECOMMENDED)"
echo "-------------------------------------------"
if command -v git-lfs &> /dev/null; then
    echo "✅ git-lfs found"
    git lfs install 2>/dev/null || true
    echo "Cloning repository from HuggingFace..."
    git clone https://huggingface.co/$MODEL_NAME $DOWNLOAD_DIR
    if [ $? -eq 0 ]; then
        echo "✅ Clone complete!"
        echo ""
        echo "To use this model, set one of the following:"
        echo "  Option 1: Set environment variable:"
        echo "    export GRANITE_MODEL_PATH=$DOWNLOAD_DIR"
        echo ""
        echo "  Option 2: Update MAMBA_VARIANTS config in steering_s6_types_granite.py:"
        echo "    'local_path': '$DOWNLOAD_DIR'"
        exit 0
    else
        echo "❌ Git clone failed"
    fi
else
    echo "❌ git-lfs not found"
    echo "   Install it with:"
    echo "     Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "     macOS: brew install git-lfs"
    echo "   Then run: git lfs install"
fi

echo ""
# Method 2: Using huggingface-cli
echo "Method 2: Using huggingface-cli"
echo "--------------------------------"
if command -v huggingface-cli &> /dev/null; then
    echo "✅ huggingface-cli found"
    echo "Downloading model to: $DOWNLOAD_DIR"
    huggingface-cli download $MODEL_NAME --local-dir $DOWNLOAD_DIR --local-dir-use-symlinks False
    if [ $? -eq 0 ]; then
        echo "✅ Download complete!"
        echo ""
        echo "To use this model, set one of the following:"
        echo "  Option 1: Set environment variable:"
        echo "    export GRANITE_MODEL_PATH=$DOWNLOAD_DIR"
        echo ""
        echo "  Option 2: Update MAMBA_VARIANTS config in steering_s6_types_granite.py:"
        echo "    'local_path': '$DOWNLOAD_DIR'"
        exit 0
    else
        echo "❌ Download failed with huggingface-cli"
    fi
else
    echo "❌ huggingface-cli not found"
    echo "   Install it with: pip install huggingface_hub[cli]"
fi

echo ""
echo "Method 3: Using Python script"
echo "-----------------------------"
cat > /tmp/download_granite.py << 'PYTHON_SCRIPT'
import os
from huggingface_hub import snapshot_download

model_name = "ibm-granite/granite-4.0-micro"
download_dir = os.path.expanduser("$DOWNLOAD_DIR")

print(f"Downloading {model_name} to {download_dir}...")
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=download_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Download complete!")
    print(f"\nTo use this model, set one of the following:")
    print(f"  Option 1: Set environment variable:")
    print(f"    export GRANITE_MODEL_PATH={download_dir}")
    print(f"\n  Option 2: Update MAMBA_VARIANTS config in steering_s6_types_granite.py:")
    print(f"    'local_path': '{download_dir}'")
except Exception as e:
    print(f"❌ Download failed: {e}")
    print(f"\nTry Method 3 (git lfs) or Method 4 (manual download)")
PYTHON_SCRIPT

python3 /tmp/download_granite.py
if [ $? -eq 0 ]; then
    rm /tmp/download_granite.py
    exit 0
fi

echo ""
echo "Method 4: Manual download instructions"
echo "----------------------------------------"
echo "1. Visit: https://huggingface.co/$MODEL_NAME"
echo "2. Click 'Files and versions' tab"
echo "3. Download all files (especially the .safetensors files)"
echo "4. Place them in: $DOWNLOAD_DIR"
echo "5. Set environment variable: export GRANITE_MODEL_PATH=$DOWNLOAD_DIR"
echo ""
echo "Required files:"
echo "  - config.json"
echo "  - tokenizer.json (or tokenizer files)"
echo "  - model-*.safetensors (model weights)"
echo "  - model.safetensors.index.json"
echo ""

