# BlackMamba Setup Instructions

## Overview
BlackMamba has been added to the steering script. The model requires the BlackMamba repository to be cloned and installed.

## Model Information
- **Model**: BlackMamba-1.5B (340M forward pass parameters, 1.5B total parameters)
- **HuggingFace**: `Zyphra/BlackMamba-1.5B`
- **GitHub**: https://github.com/Zyphra/BlackMamba
- **Paper**: https://arxiv.org/abs/2402.01771

## Setup Steps

1. **Clone the BlackMamba repository:**
   ```bash
   cd /home/vamshi
   git clone https://github.com/Zyphra/BlackMamba.git
   ```

2. **Install dependencies:**
   ```bash
   cd BlackMamba
   pip install causal-conv1d>=1.1.0  # Required for Mamba
   pip install torch packaging
   pip install .  # Install BlackMamba package
   ```

3. **Verify installation:**
   ```bash
   python -c "from mamba_model import MambaModel; print('✅ BlackMamba installed')"
   ```

4. **Run the steering script:**
   ```bash
   cd /home/vamshi/LLM_paper/post_hoc_extention_1
   python steering_s6_types_griffin.py
   ```

## Configuration

The BlackMamba model is configured in `MAMBA_VARIANTS`:
- **Variant name**: `blackmamba-340m`
- **Model name**: `Zyphra/BlackMamba-1.5B`
- **Architecture**: `blackmamba`
- **Expected layers**: 24
- **Expected hidden**: 1024

## Notes

- BlackMamba combines Mamba (SSM) with MoE (Switch Transformer)
- The model uses custom `MambaModel` class, not standard transformers
- The script will automatically detect if the repo is installed and use it
- If the repo is not found, it will provide clear error messages with setup instructions

## Troubleshooting

If you see "BlackMamba repo not found":
1. Verify the repo exists at `/home/vamshi/BlackMamba`
2. Check that `mamba_model.py` exists in the repo
3. Try installing: `cd BlackMamba && pip install .`

If you see import errors:
1. Install causal-conv1d: `pip install causal-conv1d>=1.1.0`
2. Reinstall BlackMamba: `cd BlackMamba && pip install -e .`

