import os
import secrets
import uuid
import json
import base64
import time

import torch
import numpy as np

import tornado.ioloop
import tornado.web
import tornado.escape

from eth_account import Account
from eth_account.messages import encode_defunct

from common import (
    create_deterministic_rowhash_matrix,
    create_row_from_hash,
    sha256_bytes,
    safe_allclose,
    DTYPE,
    NTYPE,
    A_TOL,
    R_TOL
)
AUTHORIZED_ADDRESS = os.getenv("AUTHORIZED_ADDRESS", "").lower()
# e.g. "0xAbCd1234..." the address derived from your private key

VERIFIER_PORT = int(os.getenv("VERIFIER_PORT", 14141))

GB = 1024**3

# In-memory sessions keyed by session_id
SESSIONS = {}
CURRENT_MEMORY = 0
MAX_MEMORY = int(os.getenv("MAX_MEMORY", 100*GB))  # 100 GB
SESSION_TIMEOUT = 300  # 5 minutes

# -------------------------------------------------------
# Merkle proof helpers
# -------------------------------------------------------

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

def check_freivals(A, B, Cr, r):
    """
    Freivalds check: verify A(B*r) == C*r without computing C= A*B fully.
    """
    # Use accumulation pattern that reduces error
    x = B.matmul(r)
    
    # Compare with residual-based approach
    check = A.matmul(x)
    
    # Kahan summation pattern for reduced error when checking difference
    residual = check - Cr
    residual_norm = torch.norm(residual)
    sum_norm = torch.norm(check) + 1e-10  # Avoid division by zero
    
    # Check both relative and absolute error
    relative_error = residual_norm / sum_norm
    absolute_error = residual_norm
    
    # For debugging: print the actual errors
    print(f"Relative error: {relative_error.item()}, Absolute error: {absolute_error.item()}")

    return safe_allclose(check, Cr, rtol=R_TOL, atol=A_TOL)

def check_row_correctness(A_row, B, claimed_row):
    """
    More numerically stable row verification for float32
    """
    # Compute product in blocks to reduce accumulation errors
    n = B.shape[1]
    local_check_row = torch.zeros(n, dtype=A_row.dtype, device=A_row.device)
    
    # Process in smaller blocks to reduce error accumulation
    block_size = min(1024, n)
    for j in range(0, n, block_size):
        j_end = min(j + block_size, n)
        local_check_row[j:j_end] = torch.matmul(A_row, B[:, j:j_end])
    
    # Check if rows match with appropriate tolerance
    return torch.allclose(local_check_row, claimed_row, rtol=R_TOL, atol=A_TOL)

# -------------------------------------------------------
# Tornado handlers
# -------------------------------------------------------

class BaseHandler(tornado.web.RequestHandler):
    def prepare(self):
        self._kill_timeout = tornado.ioloop.IOLoop.current().call_later(
            SESSION_TIMEOUT, self._kill_connection
        )
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
    
    def on_finish(self):
        if hasattr(self, "_kill_timeout"):
            tornado.ioloop.IOLoop.current().remove_timeout(self._kill_timeout)

    def _kill_connection(self):
        if not self._finished:
            self.set_status(408)
            self.finish("Request exceeded maximum duration.")

class InitHandler(BaseHandler):
    """
    1) Create a new session, generate random A,B and return session_id plus (n, master_seed).
    The coordinator will pass (n, master_seed) to the prover's /setAB.
    """
    def post(self):
        global CURRENT_MEMORY

        body = self.body_dict
        n = body.get("n", 16384)  # default 16384 if not provided
        if n > 2**18:
            n = 2**18

        # calculate cost of new session
        # n^2 * size_of_type * 2.5_matrices (A, B) + some overhead
        memory_cost = n**2 * np.dtype(NTYPE).itemsize * 3
        if CURRENT_MEMORY + memory_cost > MAX_MEMORY:
            # try to prune stale sessions
            sessions_to_delete = []
            for session_id, session_data in SESSIONS.items():
                if time.time() - session_data["start_time"] > SESSION_TIMEOUT:
                    sessions_to_delete.append(session_id)
                    CURRENT_MEMORY -= session_data["memory_cost"]
            for session_id in sessions_to_delete:
                del SESSIONS[session_id]

        # if we've still not up enough memory, error
        if CURRENT_MEMORY + memory_cost > MAX_MEMORY:
            self.write({"error": f"Memory limit exceeded, wait up to {SESSION_TIMEOUT} seconds for a session to expire"})
            return

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
            "spot_rows": None,
            "memory_cost": memory_cost,
            "start_time": time.time()
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
        r = create_row_from_hash(n, challenge_seed)
        session_data["r"] = r

        # Encode r as a base64 string to reduce truncation errors
        r_bytes = r.numpy().tobytes()
        r_b64 = base64.b64encode(r_bytes).decode()

        # Return the challenge vector as a list
        self.write({"challenge_vector": r_b64})

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
        
        # Encoding of raw buffer via base64 to reduce truncation errors
        Cr_b64 = body.get("Cr")
        Cr_bytes = base64.b64decode(Cr_b64)
        Cr_array = np.frombuffer(Cr_bytes, dtype=NTYPE)
        Cr_tensor = torch.from_numpy(Cr_array.copy())
        
        # Cr_string_list = self.body_dict.get("Cr", [])
        # Cr_list = [float(s) for s in Cr_string_list]
        # Cr_tensor = torch.tensor(Cr_list, dtype=DTYPE)

        # Cr_list = body.get("Cr")
        # if Cr_list is None:
        #    self.write({"error": "Missing Cr"})
        #    return

        session_data = SESSIONS[session_id]
        A, B, r = session_data["A"], session_data["B"], session_data["r"]
        if (A is None) or (B is None) or (r is None):
            self.write({"error": "Session missing A,B,r"})
            return


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
            row_data_b64 = row_obj.get("row_data")
            row_data_bytes = base64.b64decode(row_data_b64)
            row_data = np.frombuffer(row_data_bytes, dtype=NTYPE)
            merkle_path = row_obj.get("merkle_path")

            # Basic checks
            if row_idx is None or row_data is None or merkle_path is None:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # Convert to tensor
            row_data_tensor = torch.tensor(row_data, dtype=DTYPE)

            # 1) Merkle verify
            leaf_bytes = sha256_bytes(row_data_tensor.numpy().tobytes())
            path_ok = merkle_verify_leaf(leaf_bytes, row_idx, merkle_path, root)
            if not path_ok:
                results.append({"row_idx": row_idx, "pass": False})
                all_passed = False
                continue

            # 2) Row correctness
            row_of_A = A[row_idx, :]
            # local_check_row = row_of_A.matmul(B)
            # row_checks_ok = torch.allclose(local_check_row, row_data_tensor, rtol=R_TOL, atol=A_TOL)
            row_checks_ok = check_row_correctness(row_of_A, B, row_data_tensor)

            passed = bool(path_ok and row_checks_ok)
            if not passed:
                all_passed = False

            results.append({"row_idx": row_idx, "pass": passed})

        # delete session to free up memory
        del SESSIONS[session_id]

        self.write({
            "all_passed": all_passed,
            "results": results
        })


def make_app():
    return tornado.web.Application([
        (r"/init",            InitHandler),
        (r"/commitment",      CommitmentHandler),
        (r"/row_challenge",   RowChallengeHandler),
        (r"/multi_row_check", MultiRowCheckHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(VERIFIER_PORT)
    print(f"Verifier API running on port {VERIFIER_PORT}, dtype: {DTYPE}")
    tornado.ioloop.IOLoop.current().start()
