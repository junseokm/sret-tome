# ==============================================================================
# SReT-ToMe Inference Benchmarking & Evaluation Script
#
# Author: Junseo Kim (UTwente)
# ==============================================================================

import gc
import torch
import numpy as np
from thop import profile

def evaluate(model, dataset_loader, batch_size=128, throughput_iters=100):
    """
    Evaluates a model on Top-1 Acc, Parameter Count, FLOPs, Throughput, and Peak Activation Memory.

    Args:
        model: the model to evaluate.
        dataset_loader: the ImageNet DataLoader of PyTorch.
        batch_size: the batch size used for evaluation.
        throughput_iters: the number of iterations used for Throughput calculation.
    
    Returns:
        dict: a dictionary of name and values of the evaluated metrics.
    """
    
    model.eval() # set to evaluation mode
    torch.backends.cudnn.benchmark = True  # optimize GPU execution kernels

    # ! track token sequence length
    def simple_token_counter(module, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        tokens_in = input[0].shape[1]
        tokens_out = out_tensor.shape[1]
        print(f"[{module.__class__.__name__}] Tokens: {tokens_in} -> {tokens_out}")

    print("\n--- Tracking Token Sequence Length ---")
    handles = []
    
    # attach hooks
    for name, module in model.named_modules():
        if "Block" in str(type(module)): 
            handles.append(module.register_forward_hook(simple_token_counter))

    # use dummy image to trigger the print statements
    dummy_img = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        _ = model(dummy_img)

    for h in handles:
        h.remove()
    print("-----------------------------------------------\n")
    
   # ! peak activation memory
    dummy_tensor_pam = torch.randn(batch_size, 3, 224, 224).cuda()

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
    
    weights_memory_MB = base_memory / (1024**2)
    activations_memory_MB = (peak_memory - base_memory) / (1024**2)
    
    del dummy_tensor_pam # remove the reference variable
    gc.collect() # run the garbage collector
    torch.cuda.empty_cache() # release cached GPU memory

    # ! throughput
    tps = []
    bss = [batch_size, 64, 32, 16, 1]
    for bs in bss:
        dummy_tensor_tp = torch.randn(bs, 3, 224, 224).cuda()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros((throughput_iters, 1))

        # warm up the gpu
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_tensor_tp)

        torch.cuda.synchronize() # synchronzie cpu and gpu

        # thoughput calculation
        with torch.no_grad():
            # run multiple iterations to get the median
            for iter in range(throughput_iters):
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
    print(f"==================================================")
    print(f" Target Batch Size:      {batch_size}")
    print(f"--------------------------------------------------")
    print(f" Top-1 Accuracy:         {accuracy:.2f} %")
    print(f" Total Parameters:       {total_params / 1e6:.2f} M")
    print(f" Theoretical FLOPs:      {flops / 1e9:.2f} G")
    for i, tp in enumerate(tps):
        print(f" Throughput (BS={bss[i]}):    {tp:.2f} images/sec")
    print(f" Model Weights VRAM:     {weights_memory_MB:.2f} MB")
    print(f" Peak Activation Memory: {activations_memory_MB:.2f} MB")
    print(f"==================================================\n")

    return {
        "accuracy": accuracy,
        "params_M": total_params / 1e6,
        "flops_G": flops / 1e9,
        "throughput_bs128": tps[0], 
        "throughput_bs64": tps[1],   
        "throughput_bs32": tps[2],   
        "throughput_bs16": tps[3],   
        "throughput_bs1": tps[4],    
        "activation_mem_MB": activations_memory_MB
    }

if __name__ == "__main__":
    pass