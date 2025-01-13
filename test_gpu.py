import torch


def main():
    print("PyTorch version:", torch.__version__)

    # Check CUDA (NVIDIA GPUs)
    print("\nChecking for CUDA GPUs:")
    if torch.cuda.is_available():
        print(f"  CUDA is available with {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  No CUDA GPUs detected.")

    # Check Metal (Apple Silicon GPUs)
    print("\nChecking for Metal (MPS) GPUs:")
    if torch.backends.mps.is_available():
        print("  MPS (Metal Performance Shaders) is available.")
        print("  This is typically used on Apple Silicon.")
    else:
        print("  No MPS GPUs detected.")

    # Final Summary
    print("\nFinal Summary:")
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        print("  A GPU or accelerator is available!")
    else:
        print("  No GPUs or accelerators detected. Running on CPU.")


if __name__ == "__main__":
    main()
