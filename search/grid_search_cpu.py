# ==============================================================================
# SReT-ToMe Parameter Optimization Grid Search (CPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from eval_cpu import evaluate 
import csv
import platform
import itertools
import torch
import gc

def grid_search(csv_path="grid_search_cpu.csv"):
    """
    Performs a grid search using the 'initial_r' and 'alpha' values of the decaying token merging schedule on the SReT + ToMe architecture on a CPU.

    Args:
        csv_path: the path for the resulting csv file.
    """

    # values to explore
    initial_r_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    search_space = list(itertools.product(initial_r_ratios, alphas))
    
    columns = [
        "initial_r", "alpha", "latency", "throughput"
    ]
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    
    # run baseline
    print("BASELINE")
    base_metrics = evaluate("sret")
    
    with open(csv_path, mode="a", newline="") as f:
        csv.writer(f).writerow([
            0.0, 1.0, 
            base_metrics.get("latency", "FAILED"), 
            base_metrics.get("throughput", "FAILED"), 
        ])

    # iterate through the grid
    total_trials = len(search_space)
    for idx, (initial_r, alpha) in enumerate(search_space):
        print(f" TRIAL {idx + 1} / {total_trials}: initial_r={initial_r}, alpha={alpha}")
        
        try:
            res = evaluate("sret+tome+e", initial_r=initial_r, alpha=alpha)
            
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    initial_r, alpha, 
                    res.get("latency", "FAILED"), 
                    res.get("throughput", "FAILED")
                ])
                
            print(f"-- Success! Trial {idx + 1} logged.")
            
        except Exception as e:
            # catch hardware execution exceptions/crashes 
            print(f"-- [WARNING] Trial {idx + 1} failed execution: {e}")
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([initial_r, alpha, "FAILED", "FAILED"])
        
        gc.collect()
    print(f"\n>>> CPU Grid search completed successfully. Results saved to {csv_path}.")


if __name__ == "__main__": 
    try:
        torch.set_num_threads(4)
    except RuntimeError:
        pass

    cpu_arch = platform.machine().upper()
    is_arm = "ARM" in cpu_arch or "AARCH" in cpu_arch
    
    arch_val = platform.machine()
    cores_val = f"{torch.get_num_threads()} Thread(s)"
    
    engine_val = "ARM64" if is_arm else "x86"

    print("==================================================")
    print(f"{'Target CPU Architecture:':<32}{arch_val:>18}")
    print(f"{'Constrained Compute Cores:':<32}{cores_val:>18}")
    print(f"{'Compilation Engine:':<32}{engine_val:>18}")
    print("==================================================\n")
    
    grid_search()