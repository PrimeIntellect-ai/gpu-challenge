import hashlib
import torch
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global state (demo only!)
A = None
B = None
C = None
leaves = None       # row-hashes of C
merkle_tree = None
commitment_root = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    seed = data["seed"]  # shape (n, n)
    torch.manual_seed(seed)
    random.seed(seed)
    A = torch.randn(n, n, dtype=torch.float64)
    B = torch.randn(n, n, dtype=torch.float64)

    # Compute C
    C_t = torch.matmul(A, B)
    C = C_t.cpu()  # Store on CPU for hashing

    # Build Merkle over row-hashes
    leaves_temp = []
    for i in range(C.shape[0]):
        row_bytes = C[i,:].numpy().tobytes()
        leaves_temp.append(sha256_bytes(row_bytes))
    tree = merkle_build_tree(leaves_temp)
    root = merkle_find_root(tree)

    leaves = leaves_temp
    merkle_tree = tree
    commitment_root = root

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12121, debug=False)
