import os
import time
import secrets
import hashlib
import uuid

import numpy as np
import torch
import numba

import tornado.ioloop
import tornado.web

VERIFIER_PORT = int(os.getenv("VERIFIER_PORT", 14141))

# In-memory sessions keyed by session_id
SESSIONS = {}

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

@numba.njit
def xorshift128plus_array(n, s0, s1):
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        x = s0
        y = s1
        s0 = y
        x ^= x << 23
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        out[i] = (s1 + y) & ((1 << 64) - 1)
    return out, s0, s1

def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def merkle_verify_leaf(leaf: bytes, idx: int, path: list[str], root: bytes) -> bool:
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

def create_rowhashes(n, master_seed):
    row_hashes = []
    current_seed = master_seed
    for _ in range(n):
        row_hashes.append(sha256_bytes(current_seed))
        current_seed = sha256_bytes(current_seed)
    return row_hashes, current_seed

def create_row_from_hash(n, seed_int):
    s0 = seed_int & 0xFFFFFFFFFFFFFFF
    s1 = (seed_int >> 64) & 0xFFFFFFFFFFFFFFF
    out, _, _ = xorshift128plus_array(n, s0, s1)
    return torch.tensor(out, dtype=torch.float64) / float(1 << 64)

@timer
def create_deterministic_rowhash_matrix(n, master_seed):
    row_hashes, next_hash = create_rowhashes(n, master_seed)
    rows = []
    for i in range(n):
        seed_int = int.from_bytes(row_hashes[i], "big")
        row_data = create_row_from_hash(n, seed_int)
        rows.append(row_data)
    return torch.stack(rows), next_hash

def check_freivals(A, B, Cr, r):
    """
    Freivalds check: verify A(B*r) == C*r without computing C= A*B fully.
    """
    x = B.matmul(r)    # B*r
    check = A.matmul(x) # A * (B*r)
    return torch.allclose(check, Cr, rtol=1e-5, atol=1e-5)

# -------------------------------------------------------
# Tornado handlers
# -------------------------------------------------------

class InitHandler(tornado.web.RequestHandler):
    """
    1) Create a new session, generate random A,B and return session_id plus (n, master_seed).
    The coordinator will pass (n, master_seed) to the prover's /setAB.
    """
    def post(self):
        # If 'n' is provided, use it, else default to 16384
        n_str = self.get_argument("n", None)
        n = int(n_str) if n_str else 16384
        # set a limit on n so the process doesn't crash
        # trying to allocate huge amounts of memory
        if n > 2**20:
            n = 2**20

        master_seed = secrets.token_bytes(16)
        A, next_seed = create_deterministic_rowhash_matrix(n, master_seed)
        B, _         = create_deterministic_rowhash_matrix(n, next_seed)

        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "n": n,
            "master_seed": master_seed,
            "A": A,
            "B": B,
            "commitment_root": None,
            "r": None,
            "Cr": None,
            "spot_rows": None
        }

        # Return session info for the coordinator
        self.write({
            "session_id": session_id,
            "n": n,
            "master_seed": master_seed.hex()
        })

class CommitmentHandler(tornado.web.RequestHandler):
    """
    2) After the coordinator has told the prover to set A,B,
       the prover returns a commitment_root. The coordinator calls
       POST /commitment with { session_id, commitment_root }.

       The verifier stores commitment_root and returns the random challenge
       vector r that it wants the prover to use in computing C*r.
    """
    def post(self):
        session_id = self.get_argument("session_id", None)
        commitment_root_hex = self.get_argument("commitment_root", None)

        if not session_id or session_id not in SESSIONS:
            self.write({"error": "Invalid or missing session_id"})
            return
        if not commitment_root_hex:
            self.write({"error": "Missing commitment_root"})
            return

        session_data = SESSIONS[session_id]
        session_data["commitment_root"] = bytes.fromhex(commitment_root_hex)

        # Generate random challenge vector r
        n = session_data["n"]
        challenge_seed = secrets.token_bytes(16)
        r = create_row_from_hash(n, int.from_bytes(challenge_seed, "big"))
        session_data["r"] = r

        # Return the challenge vector as a list
        self.write({
            "challenge_vector": r.tolist()
        })

class RowChallengeHandler(tornado.web.RequestHandler):
    """
    3) The coordinator calls /row_challenge with { session_id, Cr }
       after the prover has computed C*r.

       The verifier does the Freivalds check with the stored A,B,r,
       compares to Cr. If it fails, we can either return an error or note
       that it's unverified.

       If it passes, the verifier picks row indices for a spot-check and returns them.
    """
    def post(self):
        session_id = self.get_argument("session_id", None)
        if not session_id or session_id not in SESSIONS:
            self.write({"error": "Invalid or missing session_id"})
            return

        session_data = SESSIONS[session_id]
        A   = session_data["A"]
        B   = session_data["B"]
        r   = session_data["r"]

        if (A is None) or (B is None) or (r is None):
            self.write({"error": "Session missing A,B,r"})
            return

        # Get Cr from request
        Cr_arg = self.get_argument("Cr", None)
        if not Cr_arg:
            self.write({"error": "Missing Cr"})
            return

        # Parse Cr into a float tensor
        # The coordinator might send it as JSON array; we handle comma-separated or similar
        # For simplicity assume it's a comma-separated string of floats
        # or you can adapt to whatever format is easier
        Cr_str_list = Cr_arg.split(",")
        Cr_floats = list(map(float, Cr_str_list))
        Cr = torch.tensor(Cr_floats, dtype=torch.float64)

        # Perform the Freivalds check
        freivalds_ok = check_freivals(A, B, Cr, r)
        if not freivalds_ok:
            self.write({"freivalds_ok": False, "spot_rows": []})
            return

        # If ok, pick some rows to spot check
        k = 2
        n = session_data["n"]
        chosen_rows = []
        while len(set(chosen_rows)) < k:
            chosen_rows = [secrets.randbelow(n) for _ in range(k)]

        # Store them so we know which ones we asked for
        session_data["spot_rows"] = chosen_rows

        self.write({
            "freivalds_ok": True,
            "spot_rows": chosen_rows
        })

class RowCheckHandler(tornado.web.RequestHandler):
    """
    4) The coordinator calls /row_check with { session_id, row_idx, row_data, merkle_path }.
       The verifier verifies the Merkle path, then checks row_data vs. A[row_idx]*B.

       Return True if passes, False otherwise.
    """
    def post(self):
        session_id = self.get_argument("session_id", None)
        row_idx_str = self.get_argument("row_idx", None)

        # row_data could be large, so we assume the coordinator passes it as a comma-separated string
        # or a JSON array. We'll parse accordingly.
        row_data_str = self.get_argument("row_data", None)
        merkle_path_str = self.get_argument("merkle_path", None)

        if not session_id or session_id not in SESSIONS:
            self.write({"result": False, "error": "Invalid or missing session_id"})
            return
        if not row_idx_str:
            self.write({"result": False, "error": "Missing row_idx"})
            return
        if not row_data_str:
            self.write({"result": False, "error": "Missing row_data"})
            return
        if not merkle_path_str:
            self.write({"result": False, "error": "Missing merkle_path"})
            return

        session_data = SESSIONS[session_id]
        row_idx = int(row_idx_str)

        # Convert row_data to a list of floats
        row_data_floats = list(map(float, row_data_str.split(",")))
        row_data_tensor = torch.tensor(row_data_floats, dtype=torch.float64)

        # Convert merkle_path to a list of hex
        # e.g. if user posted them joined by commas
        path_hex_list = merkle_path_str.split(",")

        # 1) Merkle verification
        leaf_bytes = sha256_bytes(row_data_tensor.numpy().tobytes())
        root = session_data["commitment_root"]
        if root is None:
            self.write({"result": False, "error": "No commitment_root in session"})
            return

        path_ok = merkle_verify_leaf(leaf_bytes, row_idx, path_hex_list, root)
        if not path_ok:
            self.write({"result": False})
            return

        # 2) Row correctness check
        A = session_data["A"]
        B = session_data["B"]
        row_of_A = A[row_idx, :]
        local_check_row = row_of_A.matmul(B)
        row_checks_ok = torch.allclose(local_check_row, row_data_tensor, rtol=1e-5, atol=1e-5)

        self.write({"result": bool(path_ok and row_checks_ok)})


class MultiRowCheckHandler(tornado.web.RequestHandler):
    """
    The coordinator calls /multi_row_check with JSON like:
    {
      "session_id": "...",
      "rows": [
        {
          "row_idx": 5,
          "row_data": [0.1, 0.2, 0.3, ...],
          "merkle_path": ["abcd...", "ef12..."]
        },
        {
          "row_idx": 11,
          "row_data": [0.5, 0.6, ...],
          "merkle_path": ["bcde...", "fa93..."]
        },
        ...
      ]
    }

    We return, for example:
    {
      "all_passed": true,
      "results": [
        { "row_idx": 5,  "pass": true  },
        { "row_idx": 11, "pass": false }
      ]
    }
    """

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        session_id = data.get("session_id", None)
        rows_info = data.get("rows", [])

        if not session_id or session_id not in SESSIONS:
            self.write({
                "all_passed": False,
                "error": "Invalid or missing session_id",
                "results": []
            })
            return

        session_data = SESSIONS[session_id]
        A = session_data["A"]
        B = session_data["B"]
        root = session_data["commitment_root"]

        if A is None or B is None or root is None:
            self.write({
                "all_passed": False,
                "error": "Missing required session data (A,B,root)",
                "results": []
            })
            return

        results = []
        all_passed = True

        for row_obj in rows_info:
            row_idx = row_obj.get("row_idx")
            row_data = row_obj.get("row_data")
            merkle_path = row_obj.get("merkle_path")

            # Basic sanity checks
            if row_idx is None or row_data is None or merkle_path is None:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # Convert row_data to torch tensor
            row_data_tensor = torch.tensor(row_data, dtype=torch.float64)

            # 1) Merkle verification
            leaf_bytes = sha256_bytes(row_data_tensor.numpy().tobytes())
            path_ok = merkle_verify_leaf(leaf_bytes, row_idx, merkle_path, root)
            if not path_ok:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # 2) Row correctness check
            row_of_A = A[row_idx, :]
            local_check_row = row_of_A.matmul(B)
            row_checks_ok = torch.allclose(local_check_row, row_data_tensor, rtol=1e-5, atol=1e-5)

            passed = bool(path_ok and row_checks_ok)
            if not passed:
                all_passed = False

            results.append({"row_idx": row_idx, "pass": passed})

        self.write({
            "all_passed": all_passed,
            "results": results
        })


def make_app():
    return tornado.web.Application([
        (r"/init",         InitHandler),
        (r"/commitment",   CommitmentHandler),
        (r"/row_challenge", RowChallengeHandler),
        (r"/row_check",    RowCheckHandler),
        (r"/multi_row_check",    MultiRowCheckHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(VERIFIER_PORT)
    print(f"Verifier API running on port {VERIFIER_PORT}")
    tornado.ioloop.IOLoop.current().start()
