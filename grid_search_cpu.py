# ==============================================================================
# SReT-ToMe Parameter Optimization Grid Search (CPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import csv
import gc
import os
import torch
import platform
import itertools
from SReT_ToMe import SReT_T_distill
from eval_cpu import evaluate 

def grid_search(csv_path="grid_search_cpu.csv"):
    """
    Performs a grid search using the 'initial_r_ratio' and 'alpha' values of the decaying token merging schedule on the SReT + ToMe architecture on a CPU.

    Args:
        csv_path: the path for the resulting csv file.
    """

    # values to explore
    initial_r_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    search_space = list(itertools.product(initial_r_ratios, alphas))
    
    columns = [
        "initial_r_ratio", "alpha", "latency", "throughput", "activation_ram_MB"
    ]
    
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
    
    # run baseline
    print("BASELINE")
    baseline_model = SReT_T_distill(pretrained=False, constant_r=0)
    checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
    baseline_model.load_state_dict(checkpoint['model'])
    baseline_model = baseline_model.eval()
    
    base_metrics = evaluate(baseline_model)
    
    with open(csv_path, mode="a", newline="") as f:
        csv.writer(f).writerow([
            0.0, 1.0, 
            base_metrics["latency"], 
            base_metrics["throughput"], 
            base_metrics["activation_ram_MB"]
        ])
    
    # clear variable and refresh garbage collector
    del baseline_model
    gc.collect()

    # iterate through the grid
    total_trials = len(search_space)
    for idx, (r_ratio, alpha) in enumerate(search_space):
        print(f" TRIAL {idx + 1} / {total_trials}: Ratio={r_ratio}, Alpha={alpha}")
        
        model = SReT_T_distill(pretrained=False, initial_r_ratio=r_ratio, alpha=alpha)
        checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model = model.eval()
        
        unique_op_id = int((r_ratio * 1000) + (alpha * 10))
        
        try:
            res = evaluate(
                model, 
                operation_label=f"r_{r_ratio}_a_{alpha}",
                operation_id=unique_op_id
            )
            
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    r_ratio, alpha, 
                    res["latency"], 
                    res["throughput"], 
                    res["activation_ram_MB"]
                ])
                
            print(f"-- Success! Trial {idx + 1} logged.")
            
        except Exception as e:
            # catch hardware execution exceptions/crashes 
            print(f"-- [WARNING] Trial {idx + 1} failed execution: {e}")
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([r_ratio, alpha, "FAILED", "FAILED", "FAILED"])
            
        finally:
            del model
            gc.collect()
            
    print(f"\n>>> CPU Grid search completed successfully. Results saved to {csv_path}.")

if __name__ == "__main__": 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # block GPU access
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    cpu_arch = platform.machine().upper()
    is_arm = "ARM" in cpu_arch or "AARCH" in cpu_arch
    
    arch_val = platform.machine()
    cores_val = f"{torch.get_num_threads()} Thread(s)"
    engine_val = "ARM64 via XNNPACK" if is_arm else "x86 via oneDNN"

    if not is_arm:
        torch.backends.mkldnn.enabled = False

    print("==================================================")
    print(f"{'Target CPU Architecture:':<32}{arch_val:>18}")
    print(f"{'Constrained Compute Cores:':<32}{cores_val:>18}")
    print(f"{'Compilation Engine:':<32}{engine_val:>18}")
    print("==================================================\n")
    
    grid_search()