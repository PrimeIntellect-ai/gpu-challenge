import os
import time
import secrets
import hashlib
import uuid
import json

import numpy as np
import torch
import numba

import tornado.ioloop
import tornado.web
import tornado.escape

from eth_account import Account
from eth_account.messages import encode_defunct

AUTHORIZED_ADDRESS = os.getenv("AUTHORIZED_ADDRESS", "").lower()
# e.g. "0xAbCd1234..." the address derived from your private key

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

class BaseHandler(tornado.web.RequestHandler):
    def prepare(self):
        # Parse JSON once, store in self.body_dict for child handlers
        if self.request.body:
            try:
                self.body_dict = tornado.escape.json_decode(self.request.body)
            except json.JSONDecodeError:
                self.set_status(400)
                self.write({"error": "Invalid JSON"})
                self.finish()
                return
        else:
            self.body_dict = {}

        # Extract signature from headers
        signature_hex = self.request.headers.get("X-Signature", "")
        if not signature_hex:
            self.set_status(401)
            self.write({"error": "Missing X-Signature header"})
            self.finish()
            return

        # Rebuild the same message the coordinator used: self.request.path + sorted JSON
        endpoint = self.request.path
        sorted_keys = sorted(self.body_dict.keys())
        sorted_data = {k: self.body_dict[k] for k in sorted_keys}
        request_data_string = json.dumps(sorted_data)

        message_str = f"{endpoint}{request_data_string}"
        message = encode_defunct(text=message_str)

        # Recover address from signature
        try:
            recovered_address = Account.recover_message(
                message, 
                signature=bytes.fromhex(signature_hex)
            )
        except:
            self.set_status(401)
            self.write({"error": "Signature recovery failed"})
            self.finish()
            return

        # Check against our authorized address
        if recovered_address.lower() != AUTHORIZED_ADDRESS:
            self.set_status(401)
            self.write({"error": "Unauthorized signer"})
            self.finish()
            return

        # If all good, proceed. Child handlers can access self.body_dict.

class InitHandler(BaseHandler):
    """
    1) Create a new session, generate random A,B and return session_id plus (n, master_seed).
    The coordinator will pass (n, master_seed) to the prover's /setAB.
    """
    def post(self):
        body = self.body_dict
        n = body.get("n", 16384)  # default 16384 if not provided
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

class CommitmentHandler(BaseHandler):
    """
    2) After the coordinator has told the prover to set A,B,
       the coordinator calls POST /commitment with { session_id, commitment_root }.
       The verifier stores commitment_root and returns the random challenge vector r.
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        commitment_root_hex = body.get("commitment_root")

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
        self.write({"challenge_vector": r.tolist()})

class RowChallengeHandler(BaseHandler):
    """
    3) The coordinator calls /row_challenge with { session_id, Cr } after the prover computed C*r.
       The verifier does the Freivalds check with stored A,B,r. If passes, picks row(s) for spot-check.
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        if not session_id or session_id not in SESSIONS:
            self.write({"error": "Invalid or missing session_id"})
            return
        
        Cr_string_list = self.body_dict.get("Cr", [])
        Cr_list = [float(s) for s in Cr_string_list]

        #Cr_list = body.get("Cr")
        # if Cr_list is None:
        #    self.write({"error": "Missing Cr"})
        #    return

        session_data = SESSIONS[session_id]
        A, B, r = session_data["A"], session_data["B"], session_data["r"]
        if (A is None) or (B is None) or (r is None):
            self.write({"error": "Session missing A,B,r"})
            return

        Cr_tensor = torch.tensor(Cr_list, dtype=torch.float64)

        # Perform Freivalds
        freivalds_ok = check_freivals(A, B, Cr_tensor, r)
        if not freivalds_ok:
            self.write({"freivalds_ok": False, "spot_rows": []})
            return

        # If ok, pick some rows to spot check
        k = 2
        n = session_data["n"]
        chosen_rows = []
        while len(set(chosen_rows)) < k:
            chosen_rows = [secrets.randbelow(n) for _ in range(k)]

        session_data["spot_rows"] = chosen_rows
        self.write({
            "freivalds_ok": True,
            "spot_rows": chosen_rows
        })

class RowCheckHandler(BaseHandler):
    """
    4) Coordinator calls /row_check with JSON:
       {
         "session_id": "...",
         "row_idx": 123,
         "row_data": [...],
         "merkle_path": [...]
       }
       Verifier checks merkle path & row correctness vs. A[row_idx]*B.
       Returns { "result": bool }.
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        row_idx = body.get("row_idx")
        row_data = body.get("row_data")
        merkle_path = body.get("merkle_path")

        if not session_id or session_id not in SESSIONS:
            self.write({"result": False, "error": "Invalid or missing session_id"})
            return
        if row_idx is None or row_data is None or merkle_path is None:
            self.write({"result": False, "error": "Missing row check data"})
            return

        session_data = SESSIONS[session_id]
        A, B, root = session_data["A"], session_data["B"], session_data["commitment_root"]
        if root is None:
            self.write({"result": False, "error": "No commitment_root in session"})
            return

        # Convert row_data to a tensor
        row_data_tensor = torch.tensor(row_data, dtype=torch.float64)

        # 1) Merkle verify
        leaf_bytes = sha256_bytes(row_data_tensor.numpy().tobytes())
        path_ok = merkle_verify_leaf(leaf_bytes, row_idx, merkle_path, root)
        if not path_ok:
            self.write({"result": False})
            return

        # 2) Check correctness
        row_of_A = A[row_idx, :]
        local_check_row = row_of_A.matmul(B)
        row_checks_ok = torch.allclose(local_check_row, row_data_tensor, rtol=1e-5, atol=1e-5)

        self.write({"result": bool(path_ok and row_checks_ok)})

class MultiRowCheckHandler(BaseHandler):
    """
    5) Coordinator calls /multi_row_check with JSON:
       {
         "session_id": "...",
         "rows": [
           {
             "row_idx": 5,
             "row_data": [...],
             "merkle_path": [...]
           },
           ...
         ]
       }
       Checks each row. Returns {
         "all_passed": bool,
         "results": [ {"row_idx": ..., "pass": bool}, ... ]
       }
    """
    def post(self):
        body = self.body_dict
        session_id = body.get("session_id")
        rows_info = body.get("rows", [])

        if not session_id or session_id not in SESSIONS:
            self.write({
                "all_passed": False,
                "error": "Invalid or missing session_id",
                "results": []
            })
            return

        session_data = SESSIONS[session_id]
        A, B, root = session_data["A"], session_data["B"], session_data["commitment_root"]
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

            # Basic checks
            if row_idx is None or row_data is None or merkle_path is None:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # Convert to tensor
            row_data_tensor = torch.tensor(row_data, dtype=torch.float64)

            # 1) Merkle verify
            leaf_bytes = sha256_bytes(row_data_tensor.numpy().tobytes())
            path_ok = merkle_verify_leaf(leaf_bytes, row_idx, merkle_path, root)
            if not path_ok:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # 2) Row correctness
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
        (r"/init",            InitHandler),
        (r"/commitment",      CommitmentHandler),
        (r"/row_challenge",   RowChallengeHandler),
        (r"/row_check",       RowCheckHandler),
        (r"/multi_row_check", MultiRowCheckHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(VERIFIER_PORT)
    print(f"Verifier API running on port {VERIFIER_PORT}, using JSON-based requests.")
    tornado.ioloop.IOLoop.current().start()
