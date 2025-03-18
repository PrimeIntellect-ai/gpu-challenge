import os
import base64

import torch
import numpy as np

import tornado.ioloop
import tornado.web
import tornado.escape

from common import (
    create_deterministic_rowhash_matrix,
    sha256_bytes,
    timer,
    DEVICE,
    DTYPE,
    NTYPE
)

PORT = int(os.getenv("PORT", 12121))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Global state
A = None
B = None
C = None
leaves = None
merkle_tree = None
commitment_root = None

# function that prints 0 the first time, then on subsequent calls, prints the time elapsed since the first call
def print_elapsed_time(msg=None, restart=False):
    if not DEBUG:
        return
    import time
    if restart:
        print_elapsed_time.start_time = time.time()
    if not hasattr(print_elapsed_time, 'start_time'):
        print_elapsed_time.start_time = time.time()
        print(msg, 0)
    else:
        print(msg, time.time() - print_elapsed_time.start_time)

def merkle_build_tree(leaves: list[bytes]) -> list[bytes]:
    level = leaves[:]
    tree = []
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if (i+1 < len(level)) else left
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
def gen_merkle_data(C):
    leaves_temp = []
    for i in range(C.shape[0]):
        row_bytes = C[i,:].cpu().numpy().tobytes()
        leaves_temp.append(sha256_bytes(row_bytes))
    tree = merkle_build_tree(leaves_temp)
    root = merkle_find_root(tree)
    return root, tree, leaves_temp

def merkle_proof_path(idx: int, leaves_list: list[bytes], tree: list[bytes]) -> list[str]:
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

        sibling_idx = current_idx ^ 1
        if sibling_idx < len(level):
            path.append(level[sibling_idx].hex())

        current_idx //= 2
        level = next_level
    return path

@timer
def cuda_sync():
    if torch.cuda.is_available() and DEBUG:
        torch.cuda.synchronize()

def block_matmul(A, B, block_size=1024):
    """
    Block matrix multiplication that reduces FP32 rounding error accumulation
    """
    n = A.shape[0]
    result = torch.zeros((n, n), dtype=A.dtype, device=A.device)
    
    # Process matrix in blocks
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            # Compute one block at a time
            block_result = torch.mm(A[i:i_end, :], B[:, j:j_end])
            result[i:i_end, j:j_end] = block_result
            
            # Optional CUDA sync after each major block to ensure precision
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    return result

@timer
def compute_C(A, B):
    """Block matrix multiplication for improved FP32 stability"""
    n = A.shape[0]
    
    # For very large matrices, use blocking
    if n > 4096:
        return block_matmul(A, B, 4096)
    else:
        # For medium matrices, use vanilla matrix multiplication with synchronization
        return torch.mm(A, B)

class SetABHandler(tornado.web.RequestHandler):
    def post(self):
        global A, B, C, leaves, merkle_tree, commitment_root
        print_elapsed_time("SetABHandler", restart=True)
        data = tornado.escape.json_decode(self.request.body)
        n = data["n"]
        master_seed = bytes.fromhex(data["seed"])
        print_elapsed_time("SetABHandler: after decoding")

        A, next_seed = create_deterministic_rowhash_matrix(n, master_seed)
        B, _ = create_deterministic_rowhash_matrix(n, next_seed)

        cuda_sync()
        C = compute_C(A, B)

        print_elapsed_time("SetABHandler: after computing C")

        commitment_root, merkle_tree, leaves = gen_merkle_data(C)

        print_elapsed_time("SetABHandler: after generating merkle data")
        self.write({"status": "ok"})

class GetCommitmentHandler(tornado.web.RequestHandler):
    def get(self):
        global commitment_root
        self.write({"commitment_root": commitment_root.hex()})

class ComputeCRHandler(tornado.web.RequestHandler):
    def post(self):
        global C
        data = tornado.escape.json_decode(self.request.body)

        # Encoding of raw buffer via base64 to reduce truncation errors
        r_b64 = data["r"]
        r_bytes = base64.b64decode(r_b64)
        r_array = np.frombuffer(r_bytes, dtype=NTYPE)
        r_t = torch.from_numpy(r_array.copy()).to(DEVICE)

        # r_t = torch.tensor(r_list, dtype=DTYPE, device=DEVICE)
        C_t = C # C.to(DEVICE)
        Cr_t = torch.matmul(C_t, r_t)
        # Cr = Cr_t.cpu().tolist()
        
        # Encode Cr to base64
        Cr_bytes = Cr_t.cpu().numpy().tobytes()
        Cr_b64 = base64.b64encode(Cr_bytes).decode()

        self.write({"Cr": Cr_b64})

class GetRowProofsHandler(tornado.web.RequestHandler):
    def post(self):
        global C, leaves, merkle_tree
        data = tornado.escape.json_decode(self.request.body)
        row_idxs = data["row_idxs"]
        rows_output = []
        for row_idx in row_idxs:
            # encode row_data to base64
            row_data = C[row_idx, :].cpu().numpy().tobytes()
            row_data_b64 = base64.b64encode(row_data).decode()
            path = merkle_proof_path(row_idx, leaves, merkle_tree)
            rows_output.append({
                "row_idx": row_idx,
                "row_data": row_data_b64,
                "merkle_path": path
            })
        self.write({"rows": rows_output})

def make_app():
    return tornado.web.Application([
        (r"/setAB", SetABHandler),
        (r"/getCommitment", GetCommitmentHandler),
        (r"/computeCR", ComputeCRHandler),
        (r"/getRowProofs", GetRowProofsHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(PORT)
    print(f"Prover API listening on port {PORT}, device: {DEVICE}, dtype: {DTYPE}")
    tornado.ioloop.IOLoop.current().start()
