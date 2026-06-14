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

Requires the official validation set of the ImageNet-1K (ILSVRC 2012) dataset. Path variable `dataset_dir` needs to be updated across scripts. 

`conda env create -f environment.yml`<br>
`conda activate sret-tome-env`<br>

## Usage

`python <eval_gpu.py | eval_cpu.py> <model_name> [--constant-r <int>] [--total-tokens <int>] [--r-ratio <float>] [--alpha <float>]`
* `<eval_gpu.py | eval_cpu.py>`: The execution environment 
* `<model_name>`: The model to evaluate (`deit`, `deit+tome+c`, `pit`, `pit+tome+c`, `pit+tome+l`, `pit+tome+e`, `sret`, `sret+tome+c`, `sret+tome+l`, `sret+tome+e`)
    - `+c` - constant reduction schedule
    - `+l` - linear reduction schedule
    - `+e` - exponential reduction schedule
* `--constant-r`: Fixed merge rate for constant reduction 
* `--total-tokens`: Total tokens to reduce for linear reduction 
* `--r-ratio`: Initial reduction percentage for exponential reduction 
* `--alpha`: Decay rate for exponential reduction

## Examples

1. DeiT Baseline Evaluation
```bash
python eval_gpu.py deit
```
```bash
==================================================
GPU:                    NVIDIA GeForce RTX 4060 Ti
==================================================

--- DeiT Baseline ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            74.40 %
Total Parameters:                           5.91 M
Theoretical FLOPs:                          2.17 G
Throughput (BS=128):            1803.92 images/sec
Throughput (BS=64):             1935.23 images/sec
Throughput (BS=32):             2021.49 images/sec
Throughput (BS=16):             2341.51 images/sec
Throughput (BS=1):               714.46 images/sec
Peak Activation Memory (BS=128):         227.20 MB
Peak Activation Memory (BS=64):          113.03 MB
Peak Activation Memory (BS=32):           56.44 MB
Peak Activation Memory (BS=16):           28.59 MB
Peak Activation Memory (BS=1):             1.76 MB
==================================================
```
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
2. PiT Constant Reduction Evaluation
```bash
python eval_gpu.py pit+tome+c --constant-r 20
```
```bash
==================================================
GPU:                    NVIDIA GeForce RTX 4060 Ti
==================================================

--- PiT + ToMe Constant Reduction Schedule  | r = 20.0 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            71.09 %
Total Parameters:                           5.10 M
Theoretical FLOPs:                          0.66 G
Throughput (BS=128):            1723.47 images/sec
Throughput (BS=64):             1784.13 images/sec
Throughput (BS=32):             1811.17 images/sec
Throughput (BS=16):             1759.97 images/sec
Throughput (BS=1):               218.20 images/sec
Peak Activation Memory (BS=128):        1193.10 MB
Peak Activation Memory (BS=64):          597.80 MB
Peak Activation Memory (BS=32):          299.43 MB
Peak Activation Memory (BS=16):          150.61 MB
Peak Activation Memory (BS=1):             9.32 MB
==================================================
```
```bash
python eval_cpu.py pit+tome+c --constant-r 20
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- PiT + ToMe Constant Reduction Schedule  | r = 20.0 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                   9.87 ms
Throughput:                         101.29 img/sec
==================================================
```
3. SReT Linear Reduction Evaluation
```bash
python eval_gpu.py sret+tome+l --total-tokens 300
```
```bash
==================================================
GPU:                    NVIDIA GeForce RTX 4060 Ti
==================================================

--- SReT + ToMe Linear Reduction Schedule | total_tokens = 300.0 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            61.36 %
Total Parameters:                           4.76 M
Theoretical FLOPs:                          1.26 G
Throughput (BS=128):            1293.42 images/sec
Throughput (BS=64):             1388.91 images/sec
Throughput (BS=32):             1421.85 images/sec
Throughput (BS=16):             1187.00 images/sec
Throughput (BS=1):               112.98 images/sec
Peak Activation Memory (BS=128):         742.90 MB
Peak Activation Memory (BS=64):          371.51 MB
Peak Activation Memory (BS=32):          188.13 MB
Peak Activation Memory (BS=16):           92.85 MB
Peak Activation Memory (BS=1):             5.80 MB
==================================================
```
```bash
python eval_cpu.py sret+tome+l --total-tokens 300
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- SReT + ToMe Linear Reduction Schedule | total_tokens = 300.0 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                  12.49 ms
Throughput:                          80.07 img/sec
==================================================
```
4. SReT Exponential Reduction Evaluation
```bash
python eval_gpu.py sret+tome+e --r-ratio 0.25 --alpha 0
```
```bash
==================================================
GPU:                    NVIDIA GeForce RTX 4060 Ti
==================================================

--- SReT + ToMe Exponential Reduction Schedule | initial_r_ratio = 0.25, alpha = 0.0 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            75.90 %
Total Parameters:                           4.76 M
Theoretical FLOPs:                          1.49 G
Throughput (BS=128):            1366.12 images/sec
Throughput (BS=64):             1493.88 images/sec
Throughput (BS=32):             1572.56 images/sec
Throughput (BS=16):             1578.22 images/sec
Throughput (BS=1):               175.68 images/sec
Peak Activation Memory (BS=128):         489.38 MB
Peak Activation Memory (BS=64):          245.46 MB
Peak Activation Memory (BS=32):          122.71 MB
Peak Activation Memory (BS=16):           62.09 MB
Peak Activation Memory (BS=1):             3.90 MB
==================================================
```
```bash
python eval_cpu.py sret+tome+e --r-ratio 0.25 --alpha 0
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- SReT + ToMe Exponential Reduction Schedule | initial_r_ratio = 0.25, alpha = 0.0 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                  11.10 ms
Throughput:                          90.12 img/sec
==================================================
```

## Code Acknowledgments & Licenses

* ToMe (Token Merging): Meta AI (CC BY-NC 4.0)
* SReT (Sliced Recursive Transformer): Zhiqiang Shen (MIT)
* PiT (Pooling-based Vision Transformer): Naver AI (Apache-2.0)
* PyTorch Image Models (timm): Ross Wightman (Apache-2.0)
* ImageNet-1K (Image Classification Dataset): Stanford Vision Lab (Custom Non-Commercial)
