#!/bin/bash
# Script to run Python with conda's Python headers while using new-env packages

# Activate new-env
source /home/vamshi/new-env/bin/activate

# Set include paths to point to conda's Python headers
export CPATH=/home/vamshi/miniconda3/envs/blackmamba/include/python3.12:${CPATH:-}
export C_INCLUDE_PATH=/home/vamshi/miniconda3/envs/blackmamba/include/python3.12:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/home/vamshi/miniconda3/envs/blackmamba/include/python3.12:${CPLUS_INCLUDE_PATH:-}

# Also set library path for linking
export LIBRARY_PATH=/home/vamshi/miniconda3/envs/blackmamba/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/home/vamshi/miniconda3/envs/blackmamba/lib:${LD_LIBRARY_PATH:-}

# Run the script with all arguments
cd /home/vamshi/LLM_paper/post_hoc_extention_1
python "$@"

