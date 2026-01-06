# M-ASK: Multi-Agent Search and Knowledge Optimization Framework

This repository contains the source code for the paper **"Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search"**.

This implementation is based on the [verl](https://github.com/volcengine/verl) library.

## Installation

Please use the following commands to set up the environment:

```
# Install dependencies (vLLM, SGLang, Megatron-Core)
bash ./scripts/install_vllm_sglang_mcore_0.7.sh

# Install M-ASK in editable mode
pip install --no-deps -e .

# Ensure numpy compatibility
pip install "numpy<2.0"
```

## Training

To start the training process, run the following script:

```
bash run_training.sh
```
