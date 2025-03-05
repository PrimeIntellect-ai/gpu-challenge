# GPU Benchmark

## Overview
This repository implements a simple **“proof-of-compute”** style benchmark that forces a remote node (e.g., a GPU worker) to perform large-scale matrix multiplications, while allowing the job originator to **verify** correctness with minimal overhead. It leverages **Freivalds’ algorithm**, which provides a highly efficient way to check whether \(C\) is truly \(A \times B\) without recomputing the full product.

## Features
1. **Deterministic Matrix Generation**  
   - Matrices $A$ and $B$ are each generated from a small integer seed.  
   - No large data transfers are needed — just the seeds.

2. **Repeated Multiplication**  
   - To increase total compute time, we can perform repeated multiplications:
     $C_1 = A \times B,\quad C_2 = C_1 \times B,\quad \ldots$
   - Each iteration forces an $\mathcal{O}(n^3)$ GPU workload.

3. **Freivalds Verification**  
   - Verifying a single multiplication $C = A \times B$ takes only $\mathcal{O}(n^2)$ operations:
     1. Randomly pick a vector $\mathbf{r}$ after $C$ is submitted.  
     2. Check $A\,(B\,\mathbf{r}) \stackrel{?}{=} C\,\mathbf{r}$.  
   - Probability of a wrong $C$ passing is very small ($\leq 1 / 2^{32}$ in practice).

## Installation
- Install PyTorch with CUDA support.
- Clone or copy these Python files into a local directory.

## Usage
1. Adjust `n`, `seed_A`, `seed_B`, and `iterations` in the code to control matrix size and repeated multiplications.
2. Run the script:
   ```bash
   python benchmark.py
   ```
3. Check the output:
   - **Freivalds check passed?** `True` means the remote node’s product is correct with high probability.
   - `False` indicates cheating or a serious numerical issue.

## Notes
- **Performance**: On a high-end GPU (e.g., H100), large \(n \times n\) multiplications can take several seconds each when \(n\) is in the tens of thousands, depending on memory bandwidth and compute capacity.
- **Floating-Point vs. Integer**: This example uses float32, but one can similarly implement integer-based matrices (mod \(2^{32}\) or mod a prime \(p\)) if desired.  
- **Dockerization**: To encapsulate this benchmark, simply include these Python files and the Dockerfile that installs PyTorch + CUDA drivers. The container then only needs to receive small integer seeds at runtime to generate and multiply the large matrices.  
- **Scaling**: If you need an even more expensive benchmark:
  - Increase `n` for greater memory impact.
  - Increase the number of repeated multiplications `iterations` for greater compute impact.

This scheme ensures the worker must do full \( \mathcal{O}(n^3) \) work while the originator expends only \( \mathcal{O}(n^2) \) time on verification, making it ideal for measuring or enforcing substantial GPU compute usage.