# ==============================================================================
# SReT-ToMe Inference Benchmarking & Evaluation Script (CPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import os

# prevent C++ errors and seg faults
os.environ["DNNL_PRIMITIVE_CACHE_CAPACITY"] = "0" 
os.environ["TORCH_DYNAMO_DISABLE"] = "1"

# restrict to 4 cores
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.profiler")
os.environ["TORCH_CPP_LOG_LEVEL"] = "FATAL" 
os.environ["GLOG_minloglevel"] = "3"

import gc
import torch
import platform
import numpy as np
import argparse

from sret.SReT import SReT_T_distill
import integration.SReT_ToMe as SReT_ToMe
import integration.PiT_ToMe as PiT_ToMe
import tome
import timm

import utilities.perf_monitor as pm

def evaluate(model_name, constant_r=0, linear_r=0, alpha=0, initial_r=0.25):
    """
    Evaluates Latency and Throughput on the CPU for a given model.
    
    Args:
        model_name: the name of the model to evaluate.
        constant_r: token merge count for constant reduction.
        linear_r: token merge count for linear reduction.
        alpha: alpha value for exponential reduction.
        initial_r: initial_r_ratio value for exponential reduction.
    
    Returns:
        dict: a dictionary of name and values of the evaluated metrics.
    """
    # force strict GPU isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # enforce PyTorch thread limits
    try:
        torch.set_num_threads(4) 
    except RuntimeError:
        pass
    
    # disable MKLDNN entirely to prevent C++ buffer overruns with ToMe
    torch.backends.mkldnn.enabled = False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sret_weights_path = os.path.join(current_dir, 'weights/SReT_T_distill.pth')

    if model_name == "deit":
        model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
    elif model_name == "deit+tome+c":
        model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
        tome.patch.timm(model, prop_attn=True)
        model.r = constant_r
    elif model_name == "pit":
        model = timm.create_model("pit_ti_distilled_224", pretrained=True)
    elif model_name == "pit+tome+c":
        model = PiT_ToMe.pit_ti_distilled(pretrained=True, schedule_type="constant", constant_r=constant_r)
    elif model_name == "pit+tome+l":
        model = PiT_ToMe.pit_ti_distilled(pretrained=True, schedule_type="linear", linear_r=linear_r)
    elif model_name == "pit+tome+e":
        model = PiT_ToMe.pit_ti_distilled(pretrained=True, schedule_type="exponential", initial_r=initial_r, alpha=alpha)
    elif model_name == "sret":
        model = SReT_T_distill(pretrained=False)
        checkpoint = torch.load(sret_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif model_name == "sret+tome+c":
        model = SReT_ToMe.SReT_T_distill(pretrained=False, schedule_type="constant", constant_r=constant_r)
        checkpoint = torch.load(sret_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif model_name == "sret+tome+l":
        model = SReT_ToMe.SReT_T_distill(pretrained=False, schedule_type="linear", linear_r=linear_r)
        checkpoint = torch.load(sret_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif model_name == "sret+tome+e":
        model = SReT_ToMe.SReT_T_distill(pretrained=False, schedule_type="exponential", initial_r=initial_r, alpha=alpha)
        checkpoint = torch.load(sret_weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    # ! latency & throughput
    latencies_MS = []
    dummy_tensor_tp = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        # warmup cpu cache
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
    median_latency_MS = float(np.median(latencies_MS))
    simulated_throughput = (1 / (median_latency_MS / 1000.0))

    print("==================================================")
    print(f"{'Target Batch Size:':<32}{1:>18}")
    print("--------------------------------------------------")
    print(f"{'Latency:':<32}{f'{median_latency_MS:.2f} ms':>18}")
    print(f"{'Throughput:':<32}{f'{simulated_throughput:.2f} img/sec':>18}")
    print("==================================================\n")
    
    del model
    del dummy_tensor_tp
    gc.collect()

    return {
        "latency": median_latency_MS,
        "throughput": simulated_throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Evaluation Script")
    parser.add_argument("model", type=str, default="deit", choices=["deit", "deit+tome+c", "pit", "pit+tome+c", "pit+tome+l", "pit+tome+e", "sret", "sret+tome+c", "sret+tome+l", "sret+tome+e"], help="Model selection (default: deit)")
    parser.add_argument("--constant-r", type=float, default=10, help="Constant token reduction rate parameter (default: 10)")
    parser.add_argument("--linear-r", type=float, default=10, help="Linear token reduction rate parameter (default: 10)")
    parser.add_argument("--initial-r", type=float, default=0.25, help="Exponential token reduction rate parameter (default: 0.25)")
    parser.add_argument("--alpha", type=float, default=0, help="Exponential token reduction rate parameter (default: 0)")
    args = parser.parse_args()

    arch_val = platform.machine()
    print("==================================================")
    print(f"{'CPU:':<32}{arch_val:>18}")
    print("==================================================\n")
    
    match (args.model):
        case "deit":
            print("--- DeiT Baseline ---")
            evaluate("deit")

        case "deit+tome+c":
            print(f"--- DeiT + ToMe Constant Reduction Schedule | constant_r = {args.constant_r} ---")
            evaluate("deit+tome+c", constant_r=args.constant_r)

        case "pit":
            print("--- PiT Baseline ---")
            evaluate("pit")

        case "pit+tome+c":
            print(f"--- PiT + ToMe Constant Reduction Schedule  | constant_r = {args.constant_r} ---")
            evaluate("pit+tome+c", constant_r=args.constant_r)

        case "pit+tome+l":
            print(f"--- PiT + ToMe Linear Reduction Schedule | linear_r = {args.linear_r} ---")
            evaluate("pit+tome+l", linear_r=args.linear_r)

        case "pit+tome+e":
            print(f"--- PiT + ToMe Exponential Reduction Schedule | initial_r = {args.initial_r}, alpha = {args.alpha} ---")
            evaluate("pit+tome+e", initial_r=args.initial_r, alpha=args.alpha)

        case "sret":
            print("--- SReT Baseline ---")
            evaluate("sret")

        case "sret+tome+c":
            print(f"--- SReT + ToMe Constant Reduction Schedule | constant_r = {args.constant_r} ---")
            evaluate("sret+tome+c", constant_r=args.constant_r)

        case "sret+tome+l":
            print(f"--- SReT + ToMe Linear Reduction Schedule | linear_r = {args.linear_r} ---")
            evaluate("sret+tome+l", linear_r=args.linear_r)

        case "sret+tome+e":
            print(f"--- SReT + ToMe Exponential Reduction Schedule | initial_r = {args.initial_r}, alpha = {args.alpha} ---")
            evaluate("sret+tome+e", initial_r=args.initial_r, alpha=args.alpha)