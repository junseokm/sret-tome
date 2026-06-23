# Combining Recursive Weight-Sharing with Token Merging for Edge Vision Transformers

University of Twente TCS Bachelor Research Project

**Author:** Junseo Kim <br>
**Supervised by:** Uraz Odyurt & Amirreza Yousefzadeh<br>


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

`python <eval_gpu.py | eval_cpu.py> <model_name> [--constant-r <int>] [--linear-r <int>] [--initial-r <float>] [--alpha <float>]`
* `<eval_gpu.py | eval_cpu.py>`: The execution environment 
* `<model_name>`: The model to evaluate (`deit`, `deit+tome+c`, `pit`, `pit+tome+c`, `pit+tome+l`, `pit+tome+e`, `sret`, `sret+tome+c`, `sret+tome+l`, `sret+tome+e`)
    - `+c` - constant reduction schedule
    - `+l` - linear reduction schedule
    - `+e` - exponential reduction schedule
* `--constant-r`: Merge rate constant reduction 
* `--linear-r`: Merge rate for linear reduction 
* `--initial-r`: Merge rate for exponential reduction 
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
Throughput (BS=128):            1808.88 images/sec
Throughput (BS=64):             1938.01 images/sec
Throughput (BS=32):             2020.95 images/sec
Throughput (BS=16):             2344.48 images/sec
Throughput (BS=1):               728.74 images/sec
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
Latency:                                   7.09 ms
Throughput:                         140.98 img/sec
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

--- PiT + ToMe Constant Reduction Schedule | constant_r = 20.0 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            71.09 %
Total Parameters:                           5.10 M
Theoretical FLOPs:                          0.66 G
Throughput (BS=128):            1729.52 images/sec
Throughput (BS=64):             1789.51 images/sec
Throughput (BS=32):             1814.40 images/sec
Throughput (BS=16):             1760.17 images/sec
Throughput (BS=1):               226.63 images/sec
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

--- PiT + ToMe Constant Reduction Schedule  | constant_r = 20.0 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                   7.55 ms
Throughput:                         132.46 img/sec
==================================================
```
3. SReT Linear Reduction Evaluation
```bash
python eval_gpu.py sret+tome+l --linear-r 10
```
```bash
==================================================
GPU:                    NVIDIA GeForce RTX 4060 Ti
==================================================

--- SReT + ToMe Linear Reduction Schedule | linear_r = 10.0 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            74.74 %
Total Parameters:                           4.76 M
Theoretical FLOPs:                          1.46 G
Throughput (BS=128):            1178.43 images/sec
Throughput (BS=64):             1280.12 images/sec
Throughput (BS=32):             1332.18 images/sec
Throughput (BS=16):             1215.14 images/sec
Throughput (BS=1):               115.80 images/sec
Peak Activation Memory (BS=128):         756.22 MB
Peak Activation Memory (BS=64):          380.27 MB
Peak Activation Memory (BS=32):          189.15 MB
Peak Activation Memory (BS=16):           94.95 MB
Peak Activation Memory (BS=1):             5.91 MB
==================================================
```
```bash
python eval_cpu.py sret+tome+l --linear-r 10
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- SReT + ToMe Linear Reduction Schedule | linear_r = 10.0 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                  14.16 ms
Throughput:                          70.64 img/sec
==================================================
```
4. SReT Exponential Reduction Evaluation
```bash
python eval_gpu.py sret+tome+e --initial-r 0.25 --alpha 0
```
```bash
==================================================
GPU:                    NVIDIA GeForce RTX 4060 Ti
==================================================

--- SReT + ToMe Exponential Reduction Schedule | initial_r = 0.25, alpha = 0.0 ---
==================================================
Target Batch Size:                             128
--------------------------------------------------
Top-1 Accuracy:                            75.96 %
Total Parameters:                           4.76 M
Theoretical FLOPs:                          1.49 G
Throughput (BS=128):            1366.58 images/sec
Throughput (BS=64):             1510.21 images/sec
Throughput (BS=32):             1588.08 images/sec
Throughput (BS=16):             1589.04 images/sec
Throughput (BS=1):               175.10 images/sec
Peak Activation Memory (BS=128):         489.45 MB
Peak Activation Memory (BS=64):          245.43 MB
Peak Activation Memory (BS=32):          123.13 MB
Peak Activation Memory (BS=16):           62.34 MB
Peak Activation Memory (BS=1):             3.90 MB
==================================================
```
```bash
python eval_cpu.py sret+tome+e --initial-r 0.25 --alpha 0
```
```bash
==================================================
CPU:                                        x86_64
==================================================

--- SReT + ToMe Exponential Reduction Schedule | initial_r = 0.25, alpha = 0.0 ---
==================================================
Target Batch Size:                               1
--------------------------------------------------
Latency:                                  11.76 ms
Throughput:                          85.05 img/sec
==================================================
```

## Code Acknowledgments & Licenses

* ToMe (Token Merging): Meta AI (CC BY-NC 4.0)
* SReT (Sliced Recursive Transformer): Zhiqiang Shen (MIT)
* PiT (Pooling-based Vision Transformer): Naver AI (Apache-2.0)
* PyTorch Image Models (timm): Ross Wightman (Apache-2.0)
* ImageNet-1K (Image Classification Dataset): Stanford Vision Lab (Custom Non-Commercial)
