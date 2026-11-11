"""Verify the environment: library versions and that PyTorch can train on the GPU.

Run with:  python -m movie_rec.check_env
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from movie_rec.config import get_device, set_seed


def main() -> None:
    print(f"Python   {sys.version.split()[0]}")
    print(f"PyTorch  {torch.__version__}")
    print(f"NumPy    {np.__version__}")
    print(f"pandas   {pd.__version__}")

    device = get_device()
    print(f"\nDevice:  {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:     {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        print(f"CUDA:    {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, training will run on the CPU.")

    set_seed()
    model = torch.nn.Sequential(
        torch.nn.Embedding(1000, 32),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randint(0, 1000, (4096, 1), device=device)
    y = torch.rand(4096, 1, device=device)

    start = time.perf_counter()
    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nSmoke test: 200 training steps in {elapsed:.2f}s, final loss {loss.item():.4f}")
    print("Environment OK.")


if __name__ == "__main__":
    main()
