# ==============================================================================
# SReT-ToMe Parameter Optimization Grid Search (GPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import csv
import gc
import torch
import itertools
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sret.SReT as SReT
from integration.SReT_ToMe import SReT_T_distill
from eval_gpu import evaluate 

def grid_search(dataset_loader, csv_path="grid_search_gpu.csv"):
    """
    Performs a grid search using the 'initial_r' and 'alpha' values of the decaying token merging schedule on the SReT + ToMe architecture on a GPU.

    Args:
        dataset_loader: the ImageNet DataLoader of PyTorch.
        csv_path: the path for the resulting csv file.
    """
   
    # values to explore
    initial_r_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    search_space = list(itertools.product(initial_r_ratios, alphas))

    sret_weights_path = os.path.join(parent_dir, 'weights/SReT_T_distill.pth')
    
    columns = [
        "initial_r", "alpha", "accuracy", "params_M", "flops_G",
        "throughput_bs128", "throughput_bs64", "throughput_bs32", 
        "throughput_bs16", "throughput_bs1", 
        "activation_mem_bs128", "activation_mem_bs64", "activation_mem_bs32",
        "activation_mem_bs16", "activation_mem_bs1"
    ]
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    
    # run baseline
    print("BASELINE")
    baseline_model = SReT.SReT_T_distill(pretrained=False)
    checkpoint = torch.load(sret_weights_path, map_location='cpu')
    baseline_model.load_state_dict(checkpoint['model'])
    baseline_model = baseline_model.cuda().eval()
    base_metrics = evaluate(baseline_model, dataset_loader)

    with open(csv_path, mode="a", newline="") as f:
        csv.writer(f).writerow([
            0.0, 1.0, base_metrics["accuracy"], base_metrics["params_M"], base_metrics["flops_G"],
            base_metrics["throughput_bs128"], base_metrics["throughput_bs64"], base_metrics["throughput_bs32"],
            base_metrics["throughput_bs16"], base_metrics["throughput_bs1"], 
            base_metrics["activation_mem_bs128"], base_metrics["activation_mem_bs64"], base_metrics["activation_mem_bs32"],
            base_metrics["activation_mem_bs16"], base_metrics["activation_mem_bs1"]
        ])
    
    # clear variable and refresh garbage collector and cache
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    # iterate through the grid
    total_trials = len(search_space)
    for idx, (initial_r, alpha) in enumerate(search_space):
        print(f" TRIAL {idx + 1} / {total_trials}: initial_r={initial_r}, alpha={alpha}")
        
        model = SReT_T_distill(pretrained=False, schedule_type="exponential", initial_r=initial_r, alpha=alpha)
        checkpoint = torch.load(sret_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.cuda().eval()
        
        try:
            res = evaluate(model, dataset_loader)
            
            # add results
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    initial_r, alpha, res["accuracy"], res["params_M"], res["flops_G"],
                    res["throughput_bs128"], res["throughput_bs64"], res["throughput_bs32"],
                    res["throughput_bs16"], res["throughput_bs1"], 
                    res["activation_mem_bs128"], res["activation_mem_bs64"], res["activation_mem_bs32"], 
                    res["activation_mem_bs16"], res["activation_mem_bs1"]  
                ])
                
            print(f"-- Success! Trial {idx + 1} logged.")
            
        except Exception as e:
            # catch hardware execution exceptions/crashes 
            print(f"-- [WARNING] Trial {idx + 1} failed execution: {e}")
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    initial_r, alpha, "FAILED", "FAILED", "FAILED",
                    "FAILED", "FAILED", "FAILED", "FAILED", "FAILED", 
                    "FAILED", "FAILED", "FAILED", "FAILED", "FAILED"
                ])
            
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
    print(f"\n>>> Grid search completed successfully. Results saved to {csv_path}.")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="GPU Grid Search Script")
    parser.add_argument("--data", type=str, default="/media/datasets/imagenet/val", help="Path to ImageNet dataset")
    args = parser.parse_args()

    dataset_dir = args.data 
    dataset_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    dataset = datasets.ImageFolder(dataset_dir, transform=dataset_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True) # batch size doesn't matter since it's only used for acc evaluations
    
    grid_search(dataset_loader)