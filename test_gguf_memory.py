#!/usr/bin/env python3
"""
Test script to load GGUF model separately and measure VRAM/RAM usage
"""
import os
import sys
import torch
import psutil
import gc
import logging
from pathlib import Path

# Add vace path to system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vace'))

# Import our GGUF loader
from vace.models.utils.gguf_loader import load_gguf_state_dict, create_gguf_model_config, dequantize_tensor, is_quantized

def get_memory_usage():
    """Get GPU and RAM memory usage"""
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        gpu_max_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    else:
        gpu_memory_allocated = gpu_memory_reserved = gpu_max_memory = 0
    
    # RAM memory
    ram_info = psutil.virtual_memory()
    ram_used = ram_info.used / 1024**3  # GB
    ram_total = ram_info.total / 1024**3  # GB
    
    return {
        'gpu_allocated': gpu_memory_allocated,
        'gpu_reserved': gpu_memory_reserved,
        'gpu_total': gpu_max_memory,
        'ram_used': ram_used,
        'ram_total': ram_total
    }

def print_memory_usage(stage):
    """Print current memory usage"""
    mem = get_memory_usage()
    print(f"\n=== {stage} ===")
    print(f"GPU Memory: {mem['gpu_allocated']:.2f}GB allocated, {mem['gpu_reserved']:.2f}GB reserved (Total: {mem['gpu_total']:.2f}GB)")
    print(f"RAM Memory: {mem['ram_used']:.2f}GB used (Total: {mem['ram_total']:.2f}GB)")

def analyze_gguf_tensors(state_dict):
    """Analyze the loaded GGUF tensors"""
    print(f"\n=== GGUF Tensor Analysis ===")
    print(f"Total tensors loaded: {len(state_dict)}")
    
    quantized_count = 0
    total_params = 0
    quantized_params = 0
    
    print("\nTensor breakdown:")
    for name, tensor in state_dict.items():
        is_quant = is_quantized(tensor)
        params = tensor.numel()
        total_params += params
        
        if is_quant:
            quantized_count += 1
            quantized_params += params
            qtype = getattr(tensor, 'tensor_type', 'unknown')
            print(f"  {name}: {tensor.shape} [{qtype}] - {params:,} params (quantized)")
        else:
            print(f"  {name}: {tensor.shape} - {params:,} params (fp16/fp32)")
    
    print(f"\nSummary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Quantized tensors: {quantized_count}/{len(state_dict)}")
    print(f"  Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.1f}%)")

def test_dequantization(state_dict, device):
    """Test dequantizing a few tensors to GPU"""
    print(f"\n=== Testing Dequantization ===")
    
    # Find a few quantized tensors to test
    quantized_tensors = [(name, tensor) for name, tensor in state_dict.items() if is_quantized(tensor)]
    
    if not quantized_tensors:
        print("No quantized tensors found!")
        return
    
    # Test dequantizing the first few
    test_count = min(3, len(quantized_tensors))
    print(f"Testing dequantization of {test_count} tensors...")
    
    for i, (name, tensor) in enumerate(quantized_tensors[:test_count]):
        print(f"\nDequantizing tensor {i+1}: {name}")
        print(f"  Original shape: {tensor.shape}")
        print(f"  Original type: {getattr(tensor, 'tensor_type', 'unknown')}")
        
        try:
            # Dequantize to GPU
            dequant_tensor = dequantize_tensor(tensor, dtype=torch.bfloat16).to(device)
            print(f"  Dequantized shape: {dequant_tensor.shape}")
            print(f"  Dequantized dtype: {dequant_tensor.dtype}")
            print(f"  GPU memory after dequant: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # Clean up
            del dequant_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ERROR dequantizing: {e}")

def main():
    print("GGUF Model Memory Test")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    
    print_memory_usage("Initial state")
    
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage("After cleanup")
    
    try:
        # GGUF model path
        gguf_path = "models/WAN2.1-VACE-14B_Q3_K_S/Wan2.1-VACE-14B-Q3_K_S.gguf"
        
        if not os.path.exists(gguf_path):
            print(f"ERROR: GGUF file not found: {gguf_path}")
            return
        
        print(f"\nLoading GGUF model from: {gguf_path}")
        file_size = Path(gguf_path).stat().st_size / 1024**3
        print(f"File size: {file_size:.2f}GB")
        
        # Load GGUF state dict
        print("\nLoading GGUF state dict...")
        state_dict = load_gguf_state_dict(gguf_path, handle_prefix="")
        
        print_memory_usage("After loading GGUF state dict")
        
        # Analyze tensors
        analyze_gguf_tensors(state_dict)
        
        # Test dequantization
        test_dequantization(state_dict, device)
        
        print_memory_usage("After dequantization tests")
        
        # Test creating model config
        print(f"\n=== Testing Model Config Creation ===")
        config = create_gguf_model_config(gguf_path)
        print("Generated config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Test moving some tensors to GPU
        print(f"\n=== Testing GPU Transfer ===")
        gpu_tensors = {}
        tensor_count = 0
        max_test_tensors = 5
        
        for name, tensor in state_dict.items():
            if tensor_count >= max_test_tensors:
                break
                
            print(f"Moving tensor to GPU: {name} {tensor.shape}")
            if is_quantized(tensor):
                # Dequantize first
                gpu_tensor = dequantize_tensor(tensor, dtype=torch.bfloat16).to(device)
            else:
                gpu_tensor = tensor.to(device)
            
            gpu_tensors[name] = gpu_tensor
            tensor_count += 1
            
            print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        
        print_memory_usage("After GPU transfer test")
        
        # Cleanup
        print("\nCleaning up...")
        del state_dict
        del gpu_tensors
        torch.cuda.empty_cache()
        gc.collect()
        
        print_memory_usage("After cleanup")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()