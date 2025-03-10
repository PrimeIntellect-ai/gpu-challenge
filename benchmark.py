import hashlib
import torch
import random

#####################################################################
# Merkle Tree Utilities
#####################################################################

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def merkle_build_tree(leaves: list[bytes]) -> list[bytes]:
    """
    Build a Merkle tree (stored in an array) from a list of leaf hashes.
    Return the array, where the tree root is at index 0.
    We'll store the tree from top to bottom for convenience:
      - tree[0] is the root
      - next level, etc.
    """
    # Start at bottom level
    level = leaves[:]  # copy
    tree = []
    # Build from bottom up
    while len(level) > 1:
        next_level = []
        # In each pass, pair up adjacent nodes
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if (i+1 < len(level)) else left
            combined = sha256_bytes(left + right)
            next_level.append(combined)
        # store current level in front (we'll reconstruct path indexing carefully)
        tree.extend(level)
        level = next_level
    # finally, the single element in 'level' is the root
    tree.extend(level)
    # The root is at the end of 'tree' as we appended up the chain
    # but let's reverse so tree[0] is the root
    tree.reverse()
    return tree

def merkle_num_leaves(leaves_count: int) -> int:
    """
    Return the number of leaves in a full binary layer that can hold leaves_count.
    This helps index the next layers properly.
    """
    # In typical Merkle usage, the tree is padded up to a power of 2
    # but here we do a simple approach that pairs the last two if needed.
    # We'll handle that in the building function. For indexing, we do something simpler:
    # We'll just store them linearly in build function. For path reconstruction, we need
    # a simpler approach or store the structure carefully. 
    # For a quick fix: we won't do an elaborate indexing trick. We'll store the entire
    # structure as done above and do a custom path approach if needed. 
    # This function might be left as a placeholder if we want a standard approach.
    # But let's just define it for completeness.
    return leaves_count

def merkle_find_root(tree: list[bytes]) -> bytes:
    """
    In our build_tree logic, tree[0] is the root.
    """
    return tree[0]

def merkle_proof_path(idx: int, leaves: list[bytes], tree: list[bytes]) -> list[bytes]:
    """
    Return the "authentication path" for leaf index `idx`.
    We do a simple top-down approach, but since we stored the entire
    Merkle structure in one array, we must carefully reconstruct the path.
    
    For demonstration, we skip a full correct indexing approach and show
    a naive approach: re-build partial trees on the fly. This is simpler
    to illustrate but not efficient for large real usage.
    """
    # We'll just rebuild from leaves each time, collecting sibling nodes.
    path = []
    level = leaves[:]
    current_idx = idx
    
    offset = 0
    # Because we appended levels in build_tree, we must replicate that:
    # We'll iteratively build next_level, then offset by len(level) in the final storage.

    # A simpler approach is to rebuild from scratch each time, ignoring 'tree' array:
    while len(level) > 1:
        # Pair up and create next level
        next_level = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i+1] if (i+1 < len(level)) else left
            combined = sha256_bytes(left + right)
            next_level.append(combined)
        # The sibling index for current_idx
        sibling_idx = current_idx ^ 1  # flip the last bit
        if sibling_idx < len(level):
            path.append(level[sibling_idx])
        # Move to the parent index
        current_idx = current_idx // 2
        level = next_level
    
    return path

def merkle_verify_leaf(leaf: bytes, idx: int, path: list[bytes], root: bytes) -> bool:
    """
    Recompute up the chain using 'path' to see if we arrive at 'root'.
    This is the standard Merkle path verification.
    """
    current = leaf
    current_idx = idx
    for sibling in path:
        if (current_idx % 2) == 0:  # even index => left child
            current = sha256_bytes(current + sibling)
        else:  # odd index => right child
            current = sha256_bytes(sibling + current)
        current_idx //= 2
    return current == root

#####################################################################
# Main Protocol Demonstration
#####################################################################

def demo_merkle_matrix_check(n=512, seed=42, device=None):
    """
    1) Verifier generates A, B (random n x n) but keeps them local.
    2) Client receives A, B, computes C = A x B.
    3) Client builds a Merkle tree of row hashes for C, returns root to verifier.
    4) Verifier sends challenge vector r, waits for C r and random row openings.
    5) Verifier checks C r with Freivalds, plus random row Merkle proofs.
    """

    # -------------
    # Setup
    # -------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dtype = torch.float64
    A = torch.randn(n, n, dtype=dtype)
    B = torch.randn(n, n, dtype=dtype)

    # -------------
    # Client side
    # -------------
    # The client receives A, B and computes C
    A_gpu = A.to(device)
    B_gpu = B.to(device)
    C_gpu = torch.matmul(A_gpu, B_gpu)
    C = C_gpu.cpu()  # shape (n, n)

    # Build a Merkle tree over row hashes of C
    leaves = []
    for row_idx in range(n):
        row_bytes = C[row_idx, :].numpy().tobytes()
        row_hash = sha256_bytes(row_bytes)
        leaves.append(row_hash)
    merkle_tree = merkle_build_tree(leaves)
    commitment_root = merkle_find_root(merkle_tree)

    # The client sends "commitment_root" to the verifier,
    # but withholds the actual data of C for now.

    # -------------
    # Verifier side
    # -------------
    # The verifier picks a random challenge vector r (kept hidden until now).
    r = torch.randn(n, dtype=dtype)

    # -------------
    # Client side
    # -------------
    # The client must now reveal C @ r, and also let the verifier open 
    # some random row(s) of C for deeper spot-check.
    Cr_gpu = torch.matmul(C_gpu, r.to(device))
    Cr = Cr_gpu.cpu()

    # Suppose we open 'k' random rows to prove they match the Merkle root
    k = 3
    chosen_rows = random.sample(range(n), k)
    openings = []
    for row_idx in chosen_rows:
        # Provide the row data
        row_data = C[row_idx, :]
        # Provide the Merkle path
        path = merkle_proof_path(row_idx, leaves, merkle_tree)
        # We'll store (row_idx, row_data, path)
        openings.append((row_idx, row_data, path))

    # -------------
    # Verifier side
    # -------------
    # 1) Verifier does Freivalds check: compare A(B r) to C r
    x = B.matmul(r)
    check = A.matmul(x)
    freivalds_ok = torch.allclose(check, Cr, rtol=1e-5, atol=1e-5)

    # 2) Verify Merkle proofs for each opened row, and optionally check that row 
    #    matches A*B for that row.
    merkle_ok = True
    row_checks_ok = True
    for (row_idx, row_data, path) in openings:
        leaf_hash = sha256_bytes(row_data.numpy().tobytes())
        # Recompute the Merkle path
        path_ok = merkle_verify_leaf(leaf_hash, row_idx, path, commitment_root)
        if not path_ok:
            merkle_ok = False
        # Optionally verify that row_data is actually row_idx of A x B:
        # row_data should be A[row_idx, :] @ B, i.e. 1 x n times n x n => 1 x n
        # We can do a small local multiplication for that single row:
        row_of_A = A[row_idx, :]  # shape [n]
        # compute row_of_A * B => shape [n],  this is O(n^2) but only for 1 row => O(n)
        local_check_row = row_of_A.matmul(B)
        if not torch.allclose(local_check_row, row_data, rtol=1e-5, atol=1e-5):
            row_checks_ok = False

    if freivalds_ok and merkle_ok and row_checks_ok:
        print("All checks passed. C is (almost certainly) correct.")
    else:
        print("Verification failed. freivalds_ok =", freivalds_ok,
              " merkle_ok =", merkle_ok, " row_checks_ok =", row_checks_ok)


if __name__ == "__main__":
    demo_merkle_matrix_check(n=16384, seed=42)
