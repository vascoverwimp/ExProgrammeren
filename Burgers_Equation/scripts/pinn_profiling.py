"""
Solving the damped harmonic oscillation
with memory and gpu usage profiling.

Usage:
    python pinn_nophys.py

Author          : Des De Borger
Email           : des.deborger@student.uantwerpen.be
Last modified   : 24/05/2026
"""

import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")  # noqa: E402

import gc
import threading
from pathlib import Path
from time import time, sleep
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from memory_profiler import memory_usage

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402

from config import Config
from data import generate_data
from model import FCNet
from plot import save_plots_from_file
from trainer import train
from utils import get_device, save_model

SCRIPT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = SCRIPT_DIR / "outputs/pinn_profiling"
OUTPUT_PATH.mkdir(exist_ok=True)


def memory_plot(
        func: callable,
        output_path: Path,
        *args,
        timestep: float = 0.01,
        ) -> None:
    """Profile and plot the RAM usage of a function over time."""
    gc.collect()
    mem = memory_usage((func, args), interval=timestep)
    mem = np.array(mem) - mem[0]  # Subtract baseline
    plt.subplots(figsize=(12, 8))
    t = np.arange(len(mem)) * timestep
    plt.plot(
        t,
        mem,
        marker='o',
        markerfacecolor='none',
        markersize=8,
        linestyle='none')
    plt.title(f"Memory usage of {func.__name__}", fontsize=20)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Used memory (in MiB)", fontsize=16)
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig(output_path, dpi=360)
    print(
        f"Memory plot generated at {output_path}."
        )
    plt.close()


def gpu_memory_plot(
        func: callable,
        output_path: Path,
        *args,
        timestep: float = 0.01,
        **kwargs
        ) -> None:
    """Profile and plot GPU VRAM usage of a function over time.    """
    samples = []
    stop_event = threading.Event()

    def sampler():
        while not stop_event.is_set():
            samples.append(torch.cuda.memory_allocated() / 1024**2)  # MiB
            sleep(timestep)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    func(*args, **kwargs)
    stop_event.set()
    thread.join()

    mem = np.array(samples)
    mem -= mem[0]  # subtract baseline
    t = np.arange(len(mem)) * timestep

    plt.subplots(figsize=(12, 8))
    plt.plot(t, mem, marker='o', markerfacecolor='none',
             markersize=8, linestyle='none')
    plt.title(f"GPU memory usage of {func.__name__}", fontsize=20)
    plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel("Used VRAM (in MiB)", fontsize=16)
    plt.ylim(bottom=0)
    plt.grid()
    plt.savefig(output_path, dpi=360)
    plt.close()
    print(f"GPU memory plot generated at {output_path}.")


def train_model(cfg: Config, data: dict) -> None:
    """Train a model"""
    device = get_device()

    # -- train standard model ------------------------------------------------
    model = FCNet(cfg)
    history, snapshots, best_state = train(
        model=model,
        data=data,
        cfg=cfg,
        device=device,
        label="Model",
        verbatim=False
    )

    # -- save model ------------------------------------------------
    save_model(
        best_state,
        history,
        snapshots,
        cfg,
        OUTPUT_PATH,
        verbatim=False
        )


def main() -> None:
    """Main loop"""
    cfg = Config()

    print('='*50)
    print('Wall time and GPU usage')
    print('='*50)

    # -- generate data --------------------------------------------
    print('Data generation profiling')
    print('='*50)
    torch.cuda.reset_peak_memory_stats()
    tic = time()
    data = generate_data(cfg)
    data_time = time() - tic
    data_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak GPU memory: {data_peak:.1f} MB")
    print(f"Wall time: {data_time:.1f} s \n")

    # -- train model ----------------------------------------------
    print('Model training profiling')
    print('='*50)
    torch.cuda.reset_peak_memory_stats()
    tic = time()
    train_model(cfg, data)
    train_time = time() - tic
    train_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak GPU memory: {train_peak:.1f} MB")
    print(f"Wall time: {train_time:.1f} s \n")

    # -- make plots ----------------------------------------------
    print('Plots saving profiling')
    print('='*50)
    torch.cuda.reset_peak_memory_stats()
    tic = time()
    save_plots_from_file(OUTPUT_PATH, verbatim=False)
    plot_time = time() - tic
    plot_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"Peak GPU memory: {plot_peak:.1f} MB")
    print(f"Wall time: {plot_time:.1f} s \n")

    print('='*50)
    print('RAM and VRAM usage over time')
    print('='*50)

    # -- generate data --------------------------------------------
    print('='*50)
    print('Data generation profiling')
    print('='*50)
    memory_plot(generate_data, OUTPUT_PATH / "mem_data.png", cfg)
    gpu_memory_plot(generate_data, OUTPUT_PATH / "mem_data_gpu.png", cfg)

    # -- train model ----------------------------------------------
    print('='*50)
    print('Model training profiling')
    print('='*50)
    torch.cuda.reset_peak_memory_stats()
    memory_plot(train_model, OUTPUT_PATH / "mem_train.png", cfg, data)
    gpu_memory_plot(train_model, OUTPUT_PATH / "mem_train_gpu.png", cfg, data)

    # -- make plots ----------------------------------------------
    print('='*50)
    print('Plots saving profiling')
    print('='*50)
    torch.cuda.reset_peak_memory_stats()
    memory_plot(
        save_plots_from_file,
        OUTPUT_PATH / "mem_plots.png",
        OUTPUT_PATH
        )
    gpu_memory_plot(
        save_plots_from_file,
        OUTPUT_PATH / "mem_plots_gpu.png",
        OUTPUT_PATH
        )


if __name__ == "__main__":
    main()
