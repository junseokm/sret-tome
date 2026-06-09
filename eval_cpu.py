# ==============================================================================
# SReT-ToMe Inference Benchmarking & Evaluation Script (Cross-Platform CPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import os
# stabilize startup behavior of C++ libraries
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = "0"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.profiler")
os.environ["TORCH_CPP_LOG_LEVEL"] = "FATAL" 
os.environ["GLOG_minloglevel"] = "3"

import gc
import torch
import platform
import numpy as np
import argparse
import multiprocessing as mp
from torch.profiler import profile, ProfilerActivity

from SReT import SReT_T_distill
import SReT_ToMe
import PiT_ToMe
import tome
import timm

import utilities.perf_monitor as pm

def isolated_worker(model_name, r, alpha, r_ratio, q):
    """
    Spawns a new process to measure peak memory and dies.

    Args:
        model_name: the name of the model to evaluate.
        r: token merge count for regular ToMe.
        alpha: alpha value for dynamic reduction.
        r_ratio: initial_r_ratio value for dynamic reduction.
        q: queue for storing the result dictionary. 
    """
    # ! force strict thread constraints
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass # ignore if the parent process already locked the threads
    
    cpu_arch = platform.machine().upper()
    is_arm = "ARM" in cpu_arch or "AARCH" in cpu_arch
    if not is_arm:
        torch.backends.mkldnn.enabled = True

    # ! instantiate the specific model 
    if model_name == "deit":
        model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
    elif model_name == "deit+tome":
        model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
        tome.patch.timm(model, prop_attn=True)
        model.r = r
    elif model_name == "pit":
        model = timm.create_model("pit_ti_distilled_224", pretrained=True)
    elif model_name == "pit+tome":
        model = PiT_ToMe.pit_ti_distilled(pretrained=True, constant_r=r)
    elif model_name == "pit+tome+d":
        model = PiT_ToMe.pit_ti_distilled(pretrained=True, initial_r_ratio=r_ratio, alpha=alpha)
    elif model_name == "sret":
        model = SReT_T_distill(pretrained=False)
        checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif model_name == "sret+tome":
        model = SReT_ToMe.SReT_T_distill(pretrained=False, constant_r=r)
        checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif model_name == "sret+tome+d":
        model = SReT_ToMe.SReT_T_distill(pretrained=False, initial_r_ratio=r_ratio, alpha=alpha)
        checkpoint = torch.load('weights/SReT_T_distill.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    # ! peak activation memory
    dummy_tensor_pam = torch.randn(1, 3, 224, 224)
    
    # warm up CPU cache
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_tensor_pam)
    gc.collect()

    # PyTorch profiling
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        with torch.no_grad():
            _ = model(dummy_tensor_pam)
            
    # get pam
    current_mem_bytes = 0
    peak_mem_bytes = 0
    
    for event in prof.events():
        current_mem_bytes += event.cpu_memory_usage

        # prevent negative memory
        current_mem_bytes = max(0, current_mem_bytes)

        if current_mem_bytes > peak_mem_bytes:
            peak_mem_bytes = current_mem_bytes
            
    activations_memory_MB = peak_mem_bytes / (1024.0 ** 2)
    
    del dummy_tensor_pam
    gc.collect()

    # ! latency & throughput
    latencies_MS = []
    dummy_tensor_tp = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        # warm up the CPU cache
        for _ in range(50):
            _ = model(dummy_tensor_tp)

        # calculate cpu time
        for _ in range(100):
            start_snapshot = pm.capture_performance()
            _ = model(dummy_tensor_tp)
            end_snapshot = pm.capture_performance()

            delta_ms = (end_snapshot.perf_time_ns - start_snapshot.perf_time_ns) / 1_000_000.0
            latencies_MS.append(delta_ms)

    # use median to get the average execution time per batch
    median_latency_MS = np.median(latencies_MS)
    simulated_throughput = (1 / (median_latency_MS / 1000.0))

    q.put({
        'latency': median_latency_MS,
        'throughput': simulated_throughput,
        'activation_ram_MB': activations_memory_MB
    })


def evaluate(model_name, r=0, alpha=0.10, r_ratio=0.30):
    """
    Main thread manager. Spawns the worker, waits for it to finish, and extracts results.

    Args:
        model_name: the name of the model to evaluate.
        r: token merge count for regular ToMe.
        alpha: alpha value for dynamic reduction.
        r_ratio: initial_r_ratio value for dynamic reduction.
    """
    q = mp.Queue()
    
    # spawn the isolated process
    p = mp.Process(target=isolated_worker, args=(model_name, r, alpha, r_ratio, q))
    p.start()
    p.join() # wait for the process to finish and die
    
    # extract the result dictionary from the queue
    results = {}
    if not q.empty():
        results = q.get()
    
    # extract metrics (default to -1 if it crashed)
    lat = results.get('latency', -1)
    tp = results.get('throughput', -1)
    mem = results.get('activation_ram_MB', -1)

    print("==================================================")
    print(f"{'Target Batch Size:':<32}{1:>18}")
    print("--------------------------------------------------")
    print(f"{'Latency:':<32}{f'{lat:.2f} ms':>18}")
    print(f"{'Throughput:':<32}{f'{tp:.2f} img/sec':>18}")
    print(f"{'True Peak Activation RAM:':<32}{f'{mem:.2f} MB':>18}")
    print("==================================================\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Evaluation Script")
    parser.add_argument("model", type=str, default="deit", choices=["deit", "deit+tome",  "pit", "pit+tome", "pit+tome+d", "sret", "sret+tome", "sret+tome+d"], help="Model selection (default: deit)")
    parser.add_argument("--alpha", type=float, default=0.10, help="Exponential token decay rate schedule modifier (default: 0.10)")
    parser.add_argument("--r-ratio", type=float, default=0.30, help="Initial token reduction percentage capability (default: 0.30)")
    args = parser.parse_args()

    arch_val = platform.machine()
    print("==================================================")
    print(f"{'CPU:':<32}{arch_val:>18}")
    print("==================================================\n")
    
    rates = [0, 10, 15, 20]
    
    match (args.model):
        case "deit":
            print("--- DeiT Baseline ---")
            evaluate("deit")

        case "deit+tome":
            for r in rates:
                print(f"--- DeiT + ToMe Baseline | r = {r} ---")
                evaluate("deit+tome", r=r)

        case "pit":
            print("--- PiT Baseline ---")
            evaluate("pit")

        case "pit+tome":
            for r in rates:
                print(f"--- PiT + ToMe Constant Reduction Baseline | r = {r} ---")
                evaluate("pit+tome", r=r)

        case "pit+tome+d":
            print(f"--- PiT + ToMe Dynamic Reduction | initial_r_ratio = {args.r_ratio}, alpha = {args.alpha} ---")
            evaluate("pit+tome+d", r_ratio=args.r_ratio, alpha=args.alpha)

        case "sret":
            print("--- SReT Baseline ---")
            evaluate("sret")

        case "sret+tome":
            for r in rates:
                print(f"--- SReT + ToMe Constant Reduction Baseline | r = {r} ---")
                evaluate("sret+tome", r=r)

        case "sret+tome+d":
            print(f"--- SReT + ToMe Dynamic Reduction | initial_r_ratio = {args.r_ratio}, alpha = {args.alpha} ---")
            evaluate("sret+tome+d", r_ratio=args.r_ratio, alpha=args.alpha)