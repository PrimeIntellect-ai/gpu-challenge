import hashlib
import torch
import fastrand
import math
import numba
import numpy as np
import time
import os
from flask import Flask, request, jsonify
from joblib import Parallel, delayed

PORT = int(os.getenv("PORT", 12121))

app = Flask(__name__)

# Global state (demo only!)
A = None
B = None
C = None
C_t = None
leaves = None       # row-hashes of C
merkle_tree = None
commitment_root = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

@timer
def build_merkle_root(C_t):
    # Move C to CPU once, convert to NumPy
    arr = C_t.cpu().numpy()

    # Hash each row in parallel exactly once
    leaves = Parallel(n_jobs=-1)(
        delayed(sha256_bytes)(arr[i].tobytes()) for i in range(arr.shape[0])
    )

    # Build the tree (for proofs) and return root
    tree = merkle_build_tree_opt(leaves)
    root = tree[0] if tree else b''
    return root, tree, leaves

@timer
def merkle_build_tree_opt(leaves: list[bytes]) -> list[bytes]:
    """
    Builds a full Merkle tree bottom-up. Returns an array with all levels,
    root at index 0. Parallelizes each level in one batch.
    """
    level = leaves
    tree = []
    while len(level) > 1:
        # Pair up
        pairs = [(level[i], level[i+1] if i+1 < len(level) else level[i])
                 for i in range(0, len(level), 2)]

        # Hash pairs in parallel once per level
        next_level = Parallel(n_jobs=-1)(
            delayed(sha256_bytes)(l + r) for (l, r) in pairs
        )

        # Store this level in `tree`
        tree.extend(level)
        level = next_level

    # Finally append the single root
    tree.extend(level)
    # Root at index 0 if reversed
    tree.reverse()
    return tree


def create_rowhashes(n, master_seed):
    row_hashes = []
    current_seed = master_seed
    for _ in range(n):
        row_hashes.append(sha256_bytes(current_seed))
        current_seed = sha256_bytes(current_seed)
    return row_hashes, current_seed

def create_row_from_hash(n, seed):
    s0 = seed & 0xFFFFFFFFFFFFFFF
    s1 = int(seed >> 64) & 0xFFFFFFFFFFFFFFF
    out, _, _ = xorshift128plus_array(n, s0, s1)
    return torch.from_numpy(out).to(device)

@timer
def create_deterministic_rowhash_matrix(n, master_seed):
    row_hashes, next_hash = create_rowhashes(n, master_seed)
    rows = []
    for i in range(n):
        row_data = create_row_from_hash(n, int.from_bytes(row_hashes[i], "big"))
        rows.append(row_data)
    return torch.stack(rows), next_hash

MASK_64 = 0xFFFFFFFFFFFFFFFF
INV_2_64 = float(2**64)

@numba.njit
def xorshift128plus_array(n, s0, s1):
    # Generate n samples. s0, s1 are 64-bit seeds.
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Xorshift128+ example
        x = s0
        y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        val = (s1 + y) & MASK_64
        out[i] = val / INV_2_64
    return out, s0, s1

#################################################################
# Merkle Tree Construction
#################################################################
def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def merkle_build_tree(leaves: list[bytes]) -> list[bytes]:
    # Build from bottom up, store all levels in an array, root at index 0.
    level = leaves[:]
    tree = []
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            if i+1 < len(level):
                right = level[i+1]
            else:
                right = left
            combined = sha256_bytes(left + right)
            next_level.append(combined)
        tree.extend(level)
        level = next_level
    tree.extend(level)
    tree.reverse()
    return tree

def merkle_find_root(tree: list[bytes]) -> bytes:
    return tree[0] if tree else b''

@timer
def compute_C(A, B):
    C_t = torch.matmul(A, B)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return C_t

@timer
def gen_merkle_data(C):
        # Build Merkle over row-hashes
    leaves_temp = []
    for i in range(C.shape[0]):
        row_bytes = C[i,:].cpu().numpy().tobytes()
        leaves_temp.append(sha256_bytes(row_bytes))
    tree = merkle_build_tree(leaves_temp)
    root = merkle_find_root(tree)
    return root, tree, leaves_temp
#################################################################
# Flask Endpoints
#################################################################

@app.route("/setAB", methods=["POST"])
def setAB():
    """
    Receive A, B in JSON form (small toy example).
    Compute C = A x B on GPU. Build a Merkle tree of row hashes for C.
    """
    global A, B, C, leaves, merkle_tree, commitment_root

    data = request.json
    n = data["n"]  # shape (n, n)
    master_seed = data["seed"]  # shape (n, n)
    master_seed = bytes.fromhex(master_seed)

    A, next_seed = create_deterministic_rowhash_matrix(n, master_seed)
    B, _ = create_deterministic_rowhash_matrix(n, next_seed)

    # Compute C
    C_t = compute_C(A, B)
    C = C_t

    # Build Merkle over row-hashes
    commitment_root, merkle_tree, leaves = gen_merkle_data(C)
    # commitment_root, merkle_tree, leaves = build_merkle_root(C_cpu)

    return jsonify({"status": "ok"})

@app.route("/getCommitment", methods=["GET"])
def getCommitment():
    """
    Return the Merkle root (commitment to C).
    """
    global commitment_root
    return jsonify({"commitment_root": commitment_root.hex()})

@app.route("/computeCR", methods=["POST"])
def computeCR():
    """
    Receive a challenge vector r, return C @ r.
    """
    global C
    data = request.json
    r_list = data["r"]  # shape (n)
    r_t = torch.tensor(r_list, dtype=torch.float64, device=device)

    C_t = C.to(device)
    Cr_t = torch.matmul(C_t, r_t)
    Cr = Cr_t.cpu().tolist()
    return jsonify({"Cr": Cr})

def merkle_proof_path(idx: int, leaves_list: list[bytes], tree: list[bytes]) -> list[str]:
    """
    Return the Merkle authentication path for a given leaf index
    (naive approach: re-build partial trees and gather sibling hashes).
    Return as list of hex strings for easy JSON transport.
    """
    path = []
    level = leaves_list[:]
    current_idx = idx
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if (i+1 < len(level)) else left
            combined = sha256_bytes(left + right)
            next_level.append(combined)

        sibling_idx = current_idx ^ 1  # flip last bit
        if sibling_idx < len(level):
            path.append(level[sibling_idx].hex())

        current_idx //= 2
        level = next_level
    return path

@app.route("/getRowProof", methods=["POST"])
def getRowProof():
    """
    Receive a row index, return:
      - row data (list of floats)
      - merkle path
    """
    global C, leaves, merkle_tree
    data = request.json
    row_idx = data["row_idx"]
    row_data = C[row_idx, :].tolist()

    # Build proof
    proof = merkle_proof_path(row_idx, leaves, merkle_tree)

    return jsonify({"row_data": row_data, "merkle_path": proof})
@app.route("/getRowProofs", methods=["POST"])
def getRowProofs():
    """
    Receives a list of row indexes. Returns an object with:
    {
      "rows": [
        {
          "row_idx": <int>,
          "row_data": [float, ...],
          "merkle_path": [str, ...]
        },
        ...
      ]
    }
    """
    global C, leaves, merkle_tree

    data = request.json
    row_idxs = data["row_idxs"]

    rows_output = []
    for row_idx in row_idxs:
        # Extract row data
        row_data = C[row_idx, :].tolist()
        # Build Merkle proof for this row
        path = merkle_proof_path(row_idx, leaves, merkle_tree)

        rows_output.append({
            "row_idx": row_idx,
            "row_data": row_data,
            "merkle_path": path
        })

    return jsonify({"rows": rows_output})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
