import requests
import torch
import hashlib
import random
import time

PORT = 12121

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def merkle_verify_leaf(leaf: bytes, idx: int, path: list[str], root: bytes) -> bool:
    """
    Recompute up the chain using 'path' (list of hex strings) to see if we arrive at 'root'.
    """
    current = leaf
    current_idx = idx
    for sibling_hex in path:
        sibling = bytes.fromhex(sibling_hex)
        if (current_idx % 2) == 0:  # even => left child
            current = sha256_bytes(current + sibling)
        else:  # odd => right child
            current = sha256_bytes(sibling + current)
        current_idx //= 2
    return current == root

def main():
    # ----------------------------
    # 1) Verifier picks random A, B, posts them to Prover
    # ----------------------------
    n = 8192  # small dimension for demonstration
    seed = torch.randint(0, 1000000, (1,)).item()
    torch.manual_seed(seed)
    dtype = torch.float64
    
    A = torch.randn(n, n, dtype=dtype)
    B = torch.randn(n, n, dtype=dtype)

    print("Sending A,B to Prover...")
    resp = requests.post(f"http://localhost:{PORT}/setAB", json={"n": n, "seed": seed})
    print("setAB response:", resp.json())

    # ----------------------------
    # 2) Get Merkle root from Prover (commitment to C)
    # ----------------------------
    resp = requests.get(f"http://localhost:{PORT}/getCommitment")
    data = resp.json()
    commitment_root_hex = data["commitment_root"]
    commitment_root = bytes.fromhex(commitment_root_hex)
    print("Commitment root (C):", commitment_root_hex)

    # ----------------------------
    # 3) Generate random challenge vector r, post to computeCR
    # ----------------------------
    r = torch.randn(n, dtype=dtype)
    r_list = r.tolist()
    resp = requests.post(f"http://localhost:{PORT}/computeCR", json={"r": r_list})
    data = resp.json()
    Cr_list = data["Cr"]
    Cr = torch.tensor(Cr_list, dtype=dtype)
    print("Received C*r from Prover.")

    # ----------------------------
    # 4) Verifier does Freivalds check: compare A(B r) with C r
    #    We can't directly do C = A*B, that costs O(n^3).
    #    Instead we do x = B*r (O(n^2)) and then A*x (O(n^2)).
    # ----------------------------
    x = B.matmul(r)
    check = A.matmul(x)
    freivalds_ok = torch.allclose(check, Cr, rtol=1e-5, atol=1e-5)
    print("Freivalds check:", freivalds_ok)

    # ----------------------------
    # 5) Spot-check a few random rows with Merkle proof
    #    We do 2 random rows
    # ----------------------------
    
    # reset seed using microsecond time so that prover can't predict the rows we'll choose
    seed = int(time.time() * 1e6)
    torch.manual_seed(seed)
    random.seed(seed)
    k = 2
    chosen_rows = random.sample(range(n), k)
    row_checks_ok = True
    merkle_ok = True
    for row_idx in chosen_rows:
        print(f"\nRequesting row {row_idx} from Prover...")
        resp = requests.post(f"http://localhost:{PORT}/getRowProof", json={"row_idx": row_idx})
        row_data = resp.json()["row_data"]
        merkle_path = resp.json()["merkle_path"]

        # 1) Merkle verification
        leaf_bytes = sha256_bytes(torch.tensor(row_data, dtype=dtype).numpy().tobytes())
        path_ok = merkle_verify_leaf(leaf_bytes, row_idx, merkle_path, commitment_root)
        if not path_ok:
            merkle_ok = False
            print("Merkle path verification failed for row", row_idx)

        # 2) Check that row_data matches row_idx of A*B (this is O(n^2) for 1 row => O(n))
        #    That's not too big, so we can do it occasionally as a spot check.
        row_of_A = A[row_idx, :]  # shape [n]
        local_check_row = row_of_A.matmul(B)  # shape [n]
        # Compare to row_data
        if not torch.allclose(local_check_row, torch.tensor(row_data, dtype=dtype), rtol=1e-5, atol=1e-5):
            row_checks_ok = False
            print("Row data mismatch at row", row_idx)
        else:
            print("Row", row_idx, "spot-check ok.")

    if freivalds_ok and merkle_ok and row_checks_ok:
        print("\nOverall: verification succeeded.")
    else:
        print("\nOverall: verification FAILED.")

if __name__ == "__main__":
    main()
