# ==============================================================================
# SReT-ToMe Inference Benchmarking & Evaluation Script (GPU)
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import os
import gc
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from thop import profile

from sret.SReT import SReT_T_distill
import integration.SReT_ToMe as SReT_ToMe
import integration.PiT_ToMe as PiT_ToMe
import tome
import timm

def evaluate(model, dataset_loader):
    """
    Evaluates a model on Top-1 Acc, Parameter Count, FLOPs, Throughput, and Peak Activation Memory on a GPU.

    Args:
        model: the model to evaluate.
        dataset_loader: the ImageNet DataLoader of PyTorch.
    
    Returns:
        dict: a dictionary of name and values of the evaluated metrics.
    """
    
    model.eval() # set to evaluation modeS
    torch.backends.cudnn.benchmark = True  # optimize GPU execution kernels
    bss = [128, 64, 32, 16, 8, 4, 2, 1] # batch sizes
    
   # ! peak activation memory
    pams = []
    for bs in bss:

        dummy_tensor_pam = torch.randn(bs, 3, 224, 224).cuda()

        with torch.no_grad():
            _ = model(dummy_tensor_pam)
        
        gc.collect() # run the garbage collector
        torch.cuda.empty_cache() # release cached GPU memory
        torch.cuda.reset_peak_memory_stats() # reset peak counter
        torch.cuda.synchronize() # synchronzie cpu and gpu
        
        base_memory = torch.cuda.memory_allocated() # get memory baseline
        
        with torch.no_grad(): # simualte inference with no gradients
            for _ in range(3): # run the model several times to see stable results
                _ = model(dummy_tensor_pam)
                
        torch.cuda.synchronize() # synchronzie cpu and gpu

        peak_memory = torch.cuda.max_memory_allocated() # get memory peak
        
        activations_memory_MB = (peak_memory - base_memory) / (1024**2)
        pams.append(activations_memory_MB)
        
        del dummy_tensor_pam # remove the reference variable
        gc.collect() # run the garbage collector
        torch.cuda.empty_cache() # release cached GPU memory

    # ! throughput
    tps = []
    for bs in bss:
        dummy_tensor_tp = torch.randn(bs, 3, 224, 224).cuda()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros((100, 1)) # run 100 iterations and get the average

        # warm up the gpu
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_tensor_tp)

        torch.cuda.synchronize() # synchronzie cpu and gpu

        # thoughput calculation
        with torch.no_grad():
            # run 100 iterations to get the average
            for iter in range(100): 
                starter.record()
                _ = model(dummy_tensor_tp)
                ender.record()
                torch.cuda.synchronize() # synchronzie cpu and gpu
                
                current_elapsed_time_MS = starter.elapsed_time(ender)
                timings[iter] = current_elapsed_time_MS

        # use median to get the average execution time per batch
        avg_batch_time_S = np.median(timings) / 1000
        # get the throughput per image
        throughput = bs / avg_batch_time_S
        tps.append(throughput)

        del dummy_tensor_tp # remove the reference variable
        gc.collect() # run the garbage collector
        torch.cuda.empty_cache() # release cached GPU memory

    # ! top-1 accuracy
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataset_loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(images)
                
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = (100 * correct / total)

    # ! parameter count and flops
    total_params = sum(p.numel() for p in model.parameters())
    
    dummy_tensor = torch.randn(1, 3, 224, 224).cuda() # batch size 1 dummy tensor
    macs, _ = profile(model, inputs=(dummy_tensor,), verbose=False)
    flops = macs * 2
    
    del dummy_tensor # remove the reference variable
    gc.collect() # run the garbage collector
    torch.cuda.empty_cache() # release cached GPU memory

    # ! results
    print("==================================================")
    print(f"{'Target Batch Size:':<32}{128:>18}")
    print("--------------------------------------------------")
    print(f"{'Top-1 Accuracy:':<32}{f'{accuracy:.2f} %':>18}")
    print(f"{'Total Parameters:':<32}{f'{total_params / 1e6:.2f} M':>18}")
    print(f"{'Theoretical FLOPs:':<32}{f'{flops / 1e9:.2f} G':>18}")
    for i, tp in enumerate(tps):
        print(f"{f'Throughput (BS={bss[i]}):':<32}{f'{tp:.2f} images/sec':>18}")
    for i, pam in enumerate(pams):
        print(f"{f'Peak Activation Memory (BS={bss[i]}):':<32}{f'{pam:.2f} MB':>18}")
    print("==================================================\n")

    return {
        "accuracy": accuracy,
        "params_M": total_params / 1e6,
        "flops_G": flops / 1e9,
        "throughput_bs128": tps[0], 
        "throughput_bs64": tps[1],   
        "throughput_bs32": tps[2],   
        "throughput_bs16": tps[3],   
        "throughput_bs8": tps[4],   
        "throughput_bs4": tps[5],   
        "throughput_bs2": tps[6],   
        "throughput_bs1": tps[7],    
        "activation_mem_bs128": pams[0],
        "activation_mem_bs64": pams[1],
        "activation_mem_bs32": pams[2],
        "activation_mem_bs16": pams[3],
        "activation_mem_bs8": pams[4],
        "activation_mem_bs4": pams[5],
        "activation_mem_bs2": pams[6],
        "activation_mem_bs1": pams[7],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Evaluation Script")
    parser.add_argument("model", type=str, default="deit", choices=["deit", "deit+tome+c", "pit", "pit+tome+c", "pit+tome+l", "pit+tome+e", "sret", "sret+tome+c", "sret+tome+l", "sret+tome+e"], help="Model selection (default: deit)")
    parser.add_argument("--constant-r", type=float, default=10, help="Constant token reduction rate parameter (default: 10)")
    parser.add_argument("--linear-r", type=float, default=10, help="Linear token reduction rate parameter (default: 10)")
    parser.add_argument("--initial-r", type=float, default=0.25, help="Exponential token reduction rate parameter (default: 0.25)")
    parser.add_argument("--alpha", type=float, default=0, help="Exponential token reduction rate parameter (default: 0)")
    parser.add_argument("--data", type=str, default="/media/datasets/imagenet/val", help="Path to ImageNet dataset")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA environment unavailable. This script must execute on a valid GPU."

    print("==================================================")
    print(f"{'GPU:':<24}{torch.cuda.get_device_name(0)}")
    print("==================================================\n")

    dataset_dir = args.data
    dataset_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(dataset_dir, transform=dataset_transform)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True) # batch size doesn't matter since it's only used for acc evaluations
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sret_weights_path = os.path.join(current_dir, 'weights/SReT_T_distill.pth')

    match (args.model):
        case "deit":
            print("--- DeiT Baseline ---")
            model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "deit+tome+c":
            print(f"--- DeiT + ToMe Constant Reduction Schedule | r = {args.constant_r} ---")
            model = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
            tome.patch.timm(model, prop_attn=True)
            model = model.cuda().eval()
            model.r = args.constant_r
            _ = evaluate(model, dataset_loader)

        case "pit":
            print("--- PiT Baseline ---")
            model = timm.create_model("pit_ti_distilled_224", pretrained=True)
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "pit+tome+c":
            print(f"--- PiT + ToMe Constant Reduction Schedule | constant_r = {args.constant_r} ---")
            model = PiT_ToMe.pit_ti_distilled(pretrained=True, schedule_type="constant", constant_r=args.constant_r)
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "pit+tome+l":
            print(f"--- PiT + ToMe Linear Reduction Schedule | linear_r = {args.linear_r} ---")
            model = PiT_ToMe.pit_ti_distilled(pretrained=True, schedule_type="linear", linear_r=args.linear_r)
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "pit+tome+e":
            print(f"--- PiT + ToMe Exponential Reduction Schedule | initial_r = {args.initial_r}, alpha = {args.alpha} ---")
            model = PiT_ToMe.pit_ti_distilled(pretrained=True, schedule_type="exponential", initial_r=args.initial_r, alpha=args.alpha)
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "sret":
            print("--- SReT Baseline ---")
            model = SReT_T_distill(pretrained=False)
            checkpoint = torch.load(sret_weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "sret+tome+c":
            print(f"--- SReT + ToMe Constant Reduction Schedule | constant_r = {args.constant_r} ---")
            model = SReT_ToMe.SReT_T_distill(pretrained=False, schedule_type="constant", constant_r=args.constant_r)
            checkpoint = torch.load(sret_weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "sret+tome+l":
            print(f"--- SReT + ToMe Linear Reduction Schedule | linear_r = {args.linear_r} ---")
            model = SReT_ToMe.SReT_T_distill(pretrained=False, schedule_type="linear", linear_r=args.linear_r)
            checkpoint = torch.load(sret_weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)

        case "sret+tome+e":
            print(f"--- SReT + ToMe Exponential Reduction Schedule | initial_r = {args.initial_r}, alpha = {args.alpha} ---")
            model = SReT_ToMe.SReT_T_distill(pretrained=False, schedule_type="exponential", initial_r=args.initial_r, alpha=args.alpha)
            checkpoint = torch.load(sret_weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model = model.cuda().eval()
            _ = evaluate(model, dataset_loader)