import torch
import sys

def check_system():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            
        # Test tensor
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print("Successfully created tensor on CUDA.")
            print(f"Tensor: {x}")
        except Exception as e:
            print(f"Failed to create tensor on CUDA: {e}")
    else:
        print("CUDA is not available. Please check your installation.")

if __name__ == "__main__":
    check_system()
