# Orthogonal Compression for Edge Vision Transformers: Recursive Weight-Sharing with Token Merging

University of Twente TCS Bachelor Research Project

## Repository Structure

```
├── images/              # Visualization images
├── logistics/           # CPU Performance Monitor helper
├── plots/               # Evaluation plots               
├── tome/                # Token Merging (ToMe) modules     
├── utilities/           # CPU Performance Monitor script            
├── weights/             # Pre-trained model checkpoints
├── PiT_ToMe.py          # PiT+ToMe integration module
├── SReT_ToMe.py         # SReT+ToMe integration module
├── SReT.py              # Original SReT 
├── eval_cpu.py          # Benchmarking script for CPU environments
├── eval_gpu.py          # Benchmarking script for GPU environments
├── grid_search_cpu.py   # Decay parameter CPU grid search script
├── grid_search_cpu.csv  # CPU grid search results
├── grid_search_gpu.py   # Decay parameter GPU grid search script
├── grid_search_gpu.csv  # GPU grid search results
├── results.ipynb        # Notebook with baseline evaluation results
├── visuals.ipynb        # Notebook for token merging visualization
├── plots.ipynb          # Notebook for plot generation
├── requirements.txt     # Python requirements
├── environment.yml      # Environment
```

## Setup

Python 3.10<br>
CUDA 13.0<br>

**GPU:** NVIDIA RTX 4060 Ti<br>
**CPU:** Intel Core Ultra 9 285K<br>

`conda env create -f environment.yml`<br>
`conda activate sret-tome-env`<br>

## Usage

`python <eval_gpu.py | eval_cpu.py> <model_name> [--r-ratio <float>] [--alpha <float>]`
* `<eval_gpu.py | eval_cpu.py>`: The execution environment **Required**
* `<model_name>`: The model to evaluate (`deit`, `deit+tome`, `pit`, `pit+tome`, `pit+tome+d`, `sret`, `sret+tome`, `sret+tome+d`) **Required**
* `--r-ratio`: Initial reduction percentage for dynamic reduction **Optional (Default: 0.30)**
* `--alpha`: Decay rate for dynamic reduction **Optional (Default: 0.10)**

## Examples
DeiT Baseline GPU Evaluation:
```bash
python eval_gpu.py deit
```
```bash
==================================================
GPU: NVIDIA GeForce RTX 4060 Ti
==================================================

--- DeiT Baseline ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            74.40 %
Total Parameters:                           5.91 M
Theoretical FLOPs:                          2.17 G
Throughput (BS=128):            1823.85 images/sec
Throughput (BS=64):             1957.84 images/sec
Throughput (BS=32):             2026.93 images/sec
Throughput (BS=16):             2369.22 images/sec
Throughput (BS=1):               732.58 images/sec
Peak Activation Memory (BS=128):         227.20 MB
Peak Activation Memory (BS=64):          113.03 MB
Peak Activation Memory (BS=32):           56.44 MB
Peak Activation Memory (BS=16):           28.59 MB
Peak Activation Memory (BS=1):             1.76 MB
==================================================
```
DeiT Baseline CPU Evaluation:
```bash
python eval_cpu.py deit
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- DeiT Baseline ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                  20.60 ms
Throughput:                          48.55 img/sec
True Peak Activation RAM:                 42.59 MB
==================================================
```
SReT+ToMe Dynamic Reduction (r-ratio=0.25, alpha=0.2) GPU Evaluation:
```bash
python eval_gpu.py sret+tome+d --r-ratio 0.25 --alpha 0.2
```
```bash
==================================================
GPU: NVIDIA GeForce RTX 4060 Ti
==================================================

--- SReT + ToMe Dynamic Reduction | initial_r_ratio = 0.25, alpha = 0.2 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            75.20 %
Total Parameters:                           4.76 M
Theoretical FLOPs:                          1.40 G
Throughput (BS=128):            1426.68 images/sec
Throughput (BS=64):             1577.19 images/sec
Throughput (BS=32):             1641.98 images/sec
Throughput (BS=16):             1632.02 images/sec
Throughput (BS=1):               143.71 images/sec
Peak Activation Memory (BS=128):         489.45 MB
Peak Activation Memory (BS=64):          245.43 MB
Peak Activation Memory (BS=32):          123.13 MB
Peak Activation Memory (BS=16):           62.34 MB
Peak Activation Memory (BS=1):             3.90 MB
==================================================
```
SReT+ToMe Dynamic Reduction (r-ratio=0.25, alpha=0.2) CPU Evaluation:
```bash
python eval_cpu.py sret+tome+d --r-ratio 0.25 --alpha 0.2
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- SReT + ToMe Dynamic Reduction | initial_r_ratio = 0.25, alpha = 0.2 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                  21.49 ms
Throughput:                          46.54 img/sec
True Peak Activation RAM:                 75.53 MB
==================================================
```

## Code Acknowledgments & Licenses

* ToMe (Token Merging): Meta AI (CC BY-NC 4.0)
* SReT (Sliced Recursive Transformer): Zhiqiang Shen (MIT)
* PiT (Pooling-based Vision Transformer): Naver AI (Apache-2.0)
* PyTorch Image Models (timm): Ross Wightman (Apache-2.0)
