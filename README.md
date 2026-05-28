# SReT-ToMe

## Orthogonal Compression for Edge Vision Transformers: Recursive Weight-Sharing with Token Merging

University of Twente TCS Bachelor Research Project

## Project Overview
TODO


## Repository Structure

```text
├── tome/                
│   ├── LICENSE          # Original ToMe license (Meta AI)
│   └── ...              # Token merging modules 
├── weights/
│   └── SReT_T_distill.pth  # Pre-trained SReT model checkpoint
├── SReT_ToMe.py         # Main module integrating SReT and ToMe
├── eval.py              # Benchmarking script for GPU environments
├── grid_search.py       # Parameter grid search script (decaying token merging schedule)
├── grid_search.csv      # Parameter grid search results
├── results.ipynb        # Notebook with baseline evaluation results
├── requirements.txt     # Complete environment blueprint for GPU/CUDA setup
└── requirements_pi.txt  # Lightweight environment blueprint for Raspberry Pi setup
```

## Environment Setup
TODO

### 1. GPU Setup (CUDA)
TODO

### 2. Raspberry Pi Setup
TODO

## Code Acknowledgments & Licenses

* **Token Merging:** Adapted directly from the official Meta AI implementation: [facebookresearch/ToMe](https://github.com/facebookresearch/ToMe). These files are in the `tome/` folder and are distributed strictly under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.
* **SReT Base Infrastructure:** Recursive transformer layers and distillation building blocks are from the Sliced Recursive Transformer framework: [ZhiqiangShen/SReT](https://www.google.com/search?q=https://github.com/ZhiqiangShen/SReT).
* **Baselines & Training Wrappers:** Baseline vision transformers, ImageNet preprocessing configurations, and tokenization utilities are supplied via the `timm` ecosystem.
