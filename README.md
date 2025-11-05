# Implementing Microsoft's 'The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits'


## Overview
- I replace every LLaMA `Linear` in **attention** and **MLP** with a custom **BitLinear** layer.
- **BitLinear** applies **RMSNorm** to inputs, then quantizes **activations** with a dynamic per-token scale (~8-bit).
- It quantizes **weights** using a mean-based scale so most values become {−1, 0, +1} (≈1-bit/ternary behavior).
- I remove the decoder layer’s `input_layernorm` because **BitLinear** already normalizes inputs.
- All other LLaMA components (attention wiring, rotary embeddings, KV heads) remain unchanged.

## Training Setup (RunPod)
- I fine-tuned on a **RunPod** GPU instance (e.g., A100) with a PyTorch+CUDA image.
- I authenticated to Weights & Biases and Hugging Face via environment variables.
- I ran the training script; runs are logged to **W&B** and checkpoints saved locally / to the **HF Hub**.

## Model Configuration
- Shapes (hidden size, heads, layers, intermediate size, context length) are set via a standard LLaMA config.
- The resulting model here is ~**225M** parameters with **BitLinear** layers.

## Results
- I was able to quantize a 8B parameter Meta LLM to a 225M parameter LLM with minimal quality loss.

## HuggingFace Deployment
https://huggingface.co/sambhav11/bitnet-llama3.1-225M

## Citation
Dettmers, Z., et al. **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.** arXiv:2402.17764. https://arxiv.org/abs/2402.17764
