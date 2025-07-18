import torch
import sys

print("=== KIỂM TRA GPU ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Test GPU computation
    print("\n=== TEST GPU COMPUTATION ===")
    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    print(f"GPU computation successful: {z.shape}")
    print(f"Device of result: {z.device}")
else:
    print("CUDA không khả dụng!")
    print("Có thể do:")
    print("1. PyTorch chưa được cài đặt với CUDA support")
    print("2. Driver NVIDIA chưa được cài đặt")
    print("3. CUDA toolkit chưa được cài đặt")

print("\n=== PYTHON PATH ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}") 