import time
import hashlib
import numpy as np
import torch
import numba
import struct

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NTYPE = np.float32

MASK_64 = 0xFFFFFFFFFFFFFFFF
MASK_32 = 0xFFFFFFFF
INV_2_64 = float(2**64)
INV_2_32 = float(2**32)

R_TOL = 5e-4
A_TOL = 1e-4

# set numba threads to max half the number of cores to avoid oversubscription
numba.set_num_threads(numba.config.NUMBA_NUM_THREADS // 2)

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

def safe_allclose(a, b, rtol=R_TOL, atol=A_TOL):
    if torch.isnan(a).any() or torch.isnan(b).any():
        return False
    if torch.isinf(a).any() or torch.isinf(b).any():
        return False
    return torch.allclose(a, b, rtol=rtol, atol=atol)

@numba.njit
def xorshift128plus_array(n, s0, s1):
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = s0
        y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        val = (s1 + y) & MASK_32
        val_float = float(val)
        out[i] = val_float * (1.0 / 4294967296.0)  # More precise than simple division
    return out, s0, s1

@numba.njit
def xorshift128plus_array64(n, s0, s1):
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = s0
        y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        val = (s1 + y) & MASK_64
        out[i] = val / INV_2_64
    return out, s0, s1

if DTYPE == torch.float32:
    xorshifter = xorshift128plus_array
else:
    xorshifter = xorshift128plus_array64

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def create_rowhashes(n, master_seed):
    hashes = []
    cur = master_seed
    for _ in range(n):
        h = sha256_bytes(cur)
        hashes.append(h)
        cur = h
    return hashes, cur

def create_row_from_hash(n, seed_hash):
    fixed_seed1 = struct.unpack("<Q", seed_hash[:8])[0]
    fixed_seed2 = struct.unpack("<Q", seed_hash[8:16])[0]
    # keep only 63 bits because numba uses signed ints
    s0 = fixed_seed1 & (2**63 - 1)
    s1 = fixed_seed2 & (2**63 - 1)
    out, _, _ = xorshifter(n, s0, s1)
    return torch.from_numpy(out)

@numba.njit(parallel=True)
def _fill_matrix(n, seeds, out_mat):
    """
    Fills 'out_mat' in-place. 
    seeds is an (n, 2) array of (s0, s1) for each row.
    """
    for i in numba.prange(n):
        s0, s1 = seeds[i, 0], seeds[i, 1]
        row_vals, _, _ = xorshifter(n, s0, s1)
        out_mat[i, :] = row_vals

@timer
def create_deterministic_rowhash_matrix(n, master_seed):
    """
    1) Generate row hashes and convert them to seeds (s0, s1).
    2) Use a Numba-parallel for loop to fill a NumPy array row-by-row.
    3) Convert to a PyTorch Tensor on the desired device.
    """
    # Prepare row hashes
    row_hashes, next_hash = create_rowhashes(n, master_seed)

    # Convert each hash into two 63-bit seeds
    seeds_np = np.empty((n, 2), dtype=np.uint64)
    for i, h in enumerate(row_hashes):
        s0 = struct.unpack("<Q", h[:8])[0] & ((1 << 63) - 1)
        s1 = struct.unpack("<Q", h[8:16])[0] & ((1 << 63) - 1)
        seeds_np[i, 0] = s0
        seeds_np[i, 1] = s1

    # Allocate the result matrix in NumPy
    mat_np = np.empty((n, n), dtype=np.float32)

    # Parallel fill
    _fill_matrix(n, seeds_np, mat_np)

    # Convert to torch Tensor and move to device
    result_torch = torch.from_numpy(mat_np).to(DEVICE)
    return result_torch, next_hash
