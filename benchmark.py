import torch

def generate_matrix(n: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """
    Deterministically generate an n x n matrix using the given seed.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    # Example: uniform in [0, 1). Adjust as desired (normal, integer, etc.).
    return torch.rand((n, n), generator=gen, dtype=torch.float32, device=device)

def repeated_matmul(A: torch.Tensor, B: torch.Tensor, iterations: int, device: str = "cuda") -> torch.Tensor:
    """
    Compute A x B, then multiply the result by B again, and so on,
    for 'iterations' times in total.

      1) C1 = A * B
      2) C2 = C1 * B
      ...
      i) Ci = Ci-1 * B

    Returns the final matrix on CPU.
    """
    A_dev = A.to(device)
    B_dev = B.to(device)

    # We'll accumulate in 'C_dev'
    C_dev = A_dev
    for _ in range(iterations):
        C_dev = torch.matmul(C_dev, B_dev)

    return C_dev.to("cpu")

def freivalds_check(C: torch.Tensor,
                    seeds: list[int], device: str = "cpu", tol: float = 1e-4) -> bool:
    """
    Freivalds’ check for verifying C ?= A x B.
    We pick a random vector r (using 'seeds'), which we do *not* reveal
    until after C is returned by the worker.
    """

    gen = torch.Generator(device=device)
    # pick random seed for r, using random function
    gen.manual_seed(torch.randint(0, 1000000, (1,)).item())

    A = generate_matrix(C.size(0), seeds[0], device=device)
    B = generate_matrix(C.size(0), seeds[1], device=device)

    # Generate a random vector of length n.
    r = torch.rand((n,), generator=gen, dtype=torch.float32, device=device)

    A_dev = A.to(device)
    B_dev = B.to(device)
    C_dev = C.to(device)

    # B*r (O(n^2)), then A*(B*r) (O(n^2)), then C*r (O(n^2))
    Br = torch.matmul(B_dev, r)
    Abr = torch.matmul(A_dev, Br)
    Cr = torch.matmul(C_dev, r)

    # Compare Abr and Cr
    return torch.allclose(Abr, Cr, rtol=tol, atol=tol)

if __name__ == "__main__":
    # Example usage:
    n = 1600        # matrix dimension
    seed_A = 123
    seed_B = 456
    iterations = 1   # number of repeated multiplications

    # Generate A, B on CPU
    A_cpu = generate_matrix(n, seed_A, device="cpu")
    B_cpu = generate_matrix(n, seed_B, device="cpu")

    # Worker side: compute repeated product
    C_cpu = repeated_matmul(A_cpu, B_cpu, iterations, device="cuda")

    # Freivalds check (the random vector seed is only revealed now)
    check_seeds = [seed_A, seed_B]
    is_correct = freivalds_check(C_cpu, check_seeds, device="cpu")
    print("Freivalds check passed?", is_correct)
