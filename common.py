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
if DTYPE == torch.float64:
    NTYPE = np.float64
R_TOL = 5e-4
A_TOL = 1e-4

MASK_64 = 0xFFFFFFFFFFFFFFFF
MASK_32 = 0xFFFFFFFF
INV_2_64 = float(2**64)
INV_2_32 = float(2**32)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

def print_with_time(msg):
    print(f"{msg} at {time.strftime('%X')}")

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
        # out[i] = val / INV_2_32
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
    row_hashes = []
    current_seed = master_seed
    for _ in range(n):
        row_hashes.append(sha256_bytes(current_seed))
        current_seed = sha256_bytes(current_seed)
    return row_hashes, current_seed

def create_row_from_hash(n, seed_hash):
    # create two 64-bit seeds from the first 16 bytes of the hash
    fixed_seed1 = struct.unpack("<Q", seed_hash[:8])[0]
    fixed_seed2 = struct.unpack("<Q", seed_hash[8:16])[0]
    # must be 63 because numba uses signed ints
    s0 = fixed_seed1 & 2**63 - 1
    s1 = fixed_seed2 & 2**63 - 1
    out, _, _ = xorshifter(n, s0, s1)
    return torch.from_numpy(out).to(DEVICE)

@timer
def create_deterministic_rowhash_matrix(n, master_seed):
    row_hashes, next_hash = create_rowhashes(n, master_seed)
    rows = []
    for i in range(n):
        row_data = create_row_from_hash(n, row_hashes[i])
        rows.append(row_data)
    return torch.stack(rows), next_hash

def safe_allclose(a, b, rtol=R_TOL, atol=A_TOL):
    """
    More robust implementation of allclose for float32
    Handles NaNs and extreme values better
    """
    # Check for NaNs first - they should never be present in valid data
    if torch.isnan(a).any() or torch.isnan(b).any():
        return False
    
    # Check for infinites - also should not be present
    if torch.isinf(a).any() or torch.isinf(b).any():
        return False
    
    # Regular comparison with tolerance
    return torch.allclose(a, b, rtol=rtol, atol=atol)