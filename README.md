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
├── SReT_ToMe.py         # Main module integrating SReT and ToMe
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
* `<eval_gpu.py | eval_cpu.py>`: The execution environment. **Required.**
* `<model_name>`: The architecture to evaluate (`deit`, `deit+tome`, `pit`, `sret`, `sret+tome`, `sret+tome+d`) **Required.**
* `--r-ratio`: Initial reduction percentage for dynamic reduction *Optional (Default: 0.30)*
* `--alpha`: Decay rate for dynamic reduction. *Optional (Default: 0.10).*

## Code Acknowledgments & Licenses

* ToMe (Token Merging): Meta AI (CC BY-NC 4.0)
* SReT (Sliced Recursive Transformer): Zhiqiang Shen (MIT)
* PyTorch Image Models (timm): Ross Wightman (Apache-2.0)
