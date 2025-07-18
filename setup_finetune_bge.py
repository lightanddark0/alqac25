#!/usr/bin/env python3
"""
Setup script for finetune_bge.py
Cài đặt nhanh tất cả dependencies cần thiết
"""

import subprocess
import sys
import os

def run_command(command):
    """Chạy lệnh và hiển thị output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success: {command}")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Error: {command}")
        print(result.stderr)
    return result.returncode == 0

def check_gpu():
    """Kiểm tra GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected, will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def main():
    print("🚀 Setting up environment for finetune_bge.py")
    print("=" * 50)
    
    # Kiểm tra Python version
    print(f"Python version: {sys.version}")
    
    # Cài đặt PyTorch với CUDA support
    print("\n📦 Installing PyTorch with CUDA support...")
    if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"):
        print("❌ Failed to install PyTorch")
        return False
    
    # Cài đặt các dependencies từ requirements file
    print("\n📦 Installing other dependencies...")
    if not run_command("pip install -r requirements_finetune_bge.txt"):
        print("❌ Failed to install dependencies")
        return False
    
    # Kiểm tra GPU
    print("\n🔍 Checking GPU...")
    gpu_available = check_gpu()
    
    # Test import các thư viện quan trọng
    print("\n🧪 Testing imports...")
    try:
        import torch
        import sentence_transformers
        import transformers
        import datasets
        import accelerate
        import faiss
        import sklearn
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test model loading
    print("\n🧪 Testing model loading...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        print("✅ Model loading successful!")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print(f"GPU available: {gpu_available}")
    print("\nYou can now run:")
    print("python finetune_bge.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 