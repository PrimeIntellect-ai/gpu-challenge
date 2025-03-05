import torch
import time

# get device from env
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

def freivalds_check(
    C: torch.Tensor,
    seeds: list[int],
    iterations: int,
    device: str = "cpu",
    tol: float = 1e-4
) -> bool:
    """
    Freivalds' check for verifying that C == A * B^iterations.

    :param C: The final n x n matrix claimed by the worker (on CPU or GPU).
    :param seeds: [seed_A, seed_B], used to deterministically generate A and B.
    :param iterations: The number of repeated multiplications. We expect C = A * (B^iterations).
    :param device: "cpu" or "cuda".
    :param tol: Tolerance for allclose().

    Steps:
      1) Generate A, B from seeds.
      2) Create a random vector r after C is received (so the worker can't cheat).
      3) Compute B^iterations * r iteratively (each iteration is O(n^2)).
      4) Multiply A * (B^iterations r) (another O(n^2)).
      5) Compare with C * r.
      6) Return True if close, else False.
    """

    # Prepare PyTorch RNG on chosen device
    gen = torch.Generator(device=device)
    # For the random vector r, we seed with some random integer:
    # (You could alternatively provide your own "secret" seed here.)
    gen.manual_seed(torch.randint(0, 1_000_000, (1,)).item())

    # Generate A, B of dimension n x n
    n = C.size(0)
    A = generate_matrix(n, seeds[0], device=device)
    B = generate_matrix(n, seeds[1], device=device)

    # Random vector r in length n
    r = torch.rand((n,), generator=gen, dtype=torch.float32, device=device)

    # We'll move everything to the chosen device
    A_dev = A.to(device)
    B_dev = B.to(device)
    C_dev = C.to(device)

    # Compute B^iterations * r iteratively
    # Start with tmp = r, then multiply by B in a loop
    tmp = r
    for _ in range(iterations):
        tmp = torch.matmul(B_dev, tmp)  # O(n^2)

    # Now tmp = B^iterations * r
    ABiter_r = torch.matmul(A_dev, tmp)  # A * (B^iterations r) => O(n^2)
    Cr = torch.matmul(C_dev, r)         # C * r => O(n^2)

    return torch.allclose(ABiter_r, Cr, rtol=tol, atol=tol)

if __name__ == "__main__":
    # Example usage:
    n = 16000        # matrix dimension
    seed_A = 123
    seed_B = 456
    iterations = 2  # number of repeated multiplications

    # Generate A, B on CPU
    start_time = time.perf_counter()
    A_cpu = generate_matrix(n, seed_A, device=DEVICE)
    B_cpu = generate_matrix(n, seed_B, device=DEVICE)
    gen_time = time.perf_counter() - start_time
    print(f"Matrix gen time for A+B took {gen_time:.2f} seconds.")

    # Worker side: compute repeated product
    start_time = time.perf_counter()
    C_cpu = repeated_matmul(A_cpu, B_cpu, iterations, device=DEVICE)
    matmul_time = time.perf_counter() - start_time
    print(f"Repeated matmul took {matmul_time:.2f} seconds.")


    # Freivalds check (the random vector seed is only revealed now)
    check_seeds = [seed_A, seed_B]
    start_time = time.perf_counter()
    is_correct = freivalds_check(C_cpu, check_seeds, iterations, device=DEVICE)
    check_time = time.perf_counter() - start_time
    print("Freivalds check passed?", is_correct)
    print(f"Verification took {check_time:.2f} seconds.")