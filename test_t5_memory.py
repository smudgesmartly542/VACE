#!/usr/bin/env python3
"""
Test script to load T5 encoder separately and measure VRAM usage
"""
import os
import sys
import torch
import psutil
import gc
from functools import partial

# Add vace path to system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vace'))

from wan.text2video import T5EncoderModel, shard_model

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

def main():
    print("T5 Encoder Memory Test")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    print(f"Using device: {device}")
    
    print_memory_usage("Initial state")
    
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage("After cleanup")
    
    try:
        # Model configuration
        checkpoint_dir = "models/WAN2.1-VACE-14B_Q3_K_S/"
        t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
        t5_tokenizer = "google/umt5-xxl"
        
        print(f"\nLoading T5 encoder from: {os.path.join(checkpoint_dir, t5_checkpoint)}")
        
        # Create T5 encoder
        shard_fn = partial(shard_model, device_id=device_id)
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,  # Force CUDA loading
            checkpoint_path=os.path.join(checkpoint_dir, t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, t5_tokenizer),
            shard_fn=None  # No sharding for single GPU
        )
        
        print_memory_usage("After T5 model creation")
        
        # Move model to GPU explicitly
        print("Moving T5 model to GPU...")
        text_encoder.model.to(device)
        
        print_memory_usage("After moving T5 to GPU")
        
        # Test encoding
        print("\nTesting text encoding...")
        test_prompt = "A sample text prompt for testing"
        with torch.no_grad():
            context = text_encoder([test_prompt], device)
            print(f"Encoded context shape: {[c.shape for c in context]}")
        
        print_memory_usage("After text encoding")
        
        # Test memory cleanup
        print("\nCleaning up...")
        del text_encoder
        del context
        torch.cuda.empty_cache()
        gc.collect()
        
        print_memory_usage("After cleanup")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()