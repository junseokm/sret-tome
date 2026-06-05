# ==============================================================================
# SReT-ToMe Inference Benchmarking & Evaluation Script (Cross-Platform CPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import os
import gc
import torch
import psutil
import platform
import numpy as np
import argparse

from SReT import SReT_T_distill
import SReT_ToMe
import tome
import timm

import utilities.perf_monitor as pm

def evaluate(model, category_label="CPU", operation_label="evaluation", operation_id=0, event_id=0):
    """
    Evaluates a model on Latency, Throughput, and Peak Activation Memory on a CPU.
    Batch size and number of threads are set to 1.

    Args:
        model: the model to evaluate.
    
    Returns:
        dict: a dictionary of name and values of the evaluated metrics.
    """

    model.eval() 
    process = psutil.Process(os.getpid())

    # ! peak activation memory
    gc.collect()
    base_memory_bytes = process.memory_info().rss 
    
    dummy_tensor_pam = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = model(dummy_tensor_pam)
    
    gc.collect()
    peak_memory_bytes = process.memory_info().rss
    
    activations_memory_MB = max(0.0, (peak_memory_bytes - base_memory_bytes) / (1024**2))
    
    del dummy_tensor_pam
    gc.collect()

    # ! latency & throughput
    latencies_MS = []
    dummy_tensor_tp = torch.randn(1, 3, 224, 224)

    # warm up the CPU cache
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_tensor_tp)

    # calculate cpu time
    with torch.no_grad(): 
        for _ in range(100): # run 100 iterations to get the average
            start_snapshot = pm.capture_performance()
            _ = model(dummy_tensor_tp)
            end_snapshot = pm.capture_performance()

            pm.record_performance_stats(
                start_snapshot=start_snapshot,
                end_snapshot=end_snapshot,
                num_cores=1, 
                category_label=category_label,
                operation_label=operation_label,
                operation_id=operation_id,
                event_id=event_id
            )
            
            delta_ms = (end_snapshot.perf_time_ns - start_snapshot.perf_time_ns) / 1_000_000.0
            latencies_MS.append(delta_ms)

    # use median to get the average execution time per batch
    median_latency_MS = np.median(latencies_MS)
    simulated_throughput = (1 / (median_latency_MS / 1000.0))

    del dummy_tensor_tp
    gc.collect()

    # ! results
    print("==================================================")
    print(f"{'Target Batch Size:':<32}{1:>18}")
    print("--------------------------------------------------")
    print(f"{'Latency:':<32}{f'{median_latency_MS:.2f} ms':>18}")
    print(f"{'Throughput:':<32}{f'{simulated_throughput:.2f} img/sec':>18}")
    print(f"{'Peak Activation RAM:':<32}{f'{activations_memory_MB:.2f} MB':>18}")
    print("==================================================\n")

    return {
        "latency": median_latency_MS,
        "throughput": simulated_throughput,
        "activation_ram_MB": activations_memory_MB,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Evaluation Script")
    parser.add_argument("model", type=str, default="deit", choices=["deit", "deit+tome", "pit", "sret", "sret+tome", "sret+tome+d"], help="Model selection (default: deit)")
    parser.add_argument("--alpha", type=float, default=0.10, help="Exponential token decay rate schedule modifier (default: 0.10)")
    parser.add_argument("--r-ratio", type=float, default=0.30, help="Initial token reduction percentage capability (default: 0.30)")
    args = parser.parse_args()

    # architecture initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # block GPU execution
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    cpu_arch = platform.machine().upper()
    is_arm = "ARM" in cpu_arch or "AARCH" in cpu_arch
    
    arch_val = platform.machine()
    cores_val = f"{torch.get_num_threads()} Thread(s)"
    engine_val = "ARM64 via XNNPACK" if is_arm else "x86 via oneDNN"

    if not is_arm:
        torch.backends.mkldnn.enabled = True

    print("==================================================")
    print(f"{'Target CPU Architecture:':<32}{arch_val:>18}")
    print(f"{'Constrained Compute Cores:':<32}{cores_val:>18}")
    print(f"{'Compilation Engine:':<32}{engine_val:>18}")
    print("==================================================\n")
    
    rates = [0, 10, 15, 20]
    match (args.model):
        case "deit":
            print("--- DeiT Baseline ---")
            model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
            _ = evaluate(model)

        case "deit+tome":
            model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
            tome.patch.timm(model, prop_attn=True)
            for r in rates:
                print(f"--- DeiT + ToMe Baseline | r = {r} ---")
                model.r = r
                _ = evaluate(model)

        case "pit":
            print("--- PiT Baseline ---")
            model = timm.create_model("pit_ti_distilled_224", pretrained=True)
            _ = evaluate(model)

        case "sret":
            print("--- SReT Baseline ---")
            model = SReT_T_distill(pretrained=False)
            checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            _ = evaluate(model)

        case "sret+tome":
            for r in rates:
                print(f"--- SReT + ToMe Constant Reduction Baseline | r = {r} ---")
                model = SReT_ToMe.SReT_T_distill(pretrained=False, constant_r=r)
                checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
                model.load_state_dict(checkpoint['model'])
                _ = evaluate(model)

        case "sret+tome+d":
            print(f"--- SReT + ToMe Dynamic Reduction | initial_r_ratio = {args.r_ratio}, alpha = {args.alpha} ---")
            model = SReT_ToMe.SReT_T_distill(pretrained=False, initial_r_ratio=args.r_ratio, alpha=args.alpha)
            checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            _ = evaluate(model)