import requests
import os
import json
import time
from eth_account import Account
from eth_account.messages import encode_defunct

VERIFIER_URL = os.getenv("VERIFIER_URL", "http://localhost:14141")
PROVER_URL   = os.getenv("PROVER_URL", "http://localhost:12121")

PRIVATE_KEY_HEX = os.getenv("PRIVATE_KEY_HEX")

# create timer decorator that prints the first string argument of the inner function as a string
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}{args[0]} took {end - start} seconds")
        return result
    return wrapper

def sign_request(endpoint: str, data: dict | None, private_key_hex: str) -> str:
    """
    Matches the Rust logic:
      1) Sort JSON keys
      2) Convert to string
      3) message = endpoint + request_data_string
      4) EIP-191 sign (personal sign)
    Returns signature hex.
    """
    if data:
        # Sort the dict by keys
        sorted_keys = sorted(data.keys())
        sorted_data = {k: data[k] for k in sorted_keys}
        request_data_string = json.dumps(sorted_data)
    else:
        request_data_string = ""

    message_str = f"{endpoint}{request_data_string}"
    message = encode_defunct(text=message_str)
    signed = Account.sign_message(message, private_key=private_key_hex)
    return signed.signature.hex()

@timer
def post_signed_json(endpoint: str, data: dict | None = None) -> requests.Response:
    """
    1) Sort the data’s keys (like your Rust code does).
    2) Convert the sorted data to JSON.
    3) Create message_str = endpoint + sorted_json.
    4) EIP-191 sign that string (personal_sign).
    5) Send the result to the verifier with:
       - 'Content-Type: application/json'
       - 'X-Signature: <hex signature>'
       - 'X-Endpoint: <endpoint>'
    6) Return the requests.Response for further processing.
    """
    # 1) Sort the data
    data = data or {}
    sorted_keys = sorted(data.keys())
    sorted_dict = {k: data[k] for k in sorted_keys}

    # 2) Convert to JSON bytes
    sorted_json = json.dumps(sorted_dict)

    # 3) Concat endpoint + sorted_json
    message_str = f"{endpoint}{sorted_json}"

    # 4) EIP-191 sign
    message = encode_defunct(text=message_str)
    signed = Account.sign_message(message, private_key=PRIVATE_KEY_HEX)
    signature_hex = signed.signature.hex()

    # 5) POST to the verifier with the signature in headers
    url = VERIFIER_URL + endpoint
    headers = {
        "Content-Type": "application/json",
        "X-Signature": signature_hex,
    }

    return requests.post(url, json=sorted_dict, headers=headers)

@timer
def post_to_prover(endpoint: str, json: dict | None = None) -> requests.Response:
    return requests.post(f"{PROVER_URL}{endpoint}", json=json).json()

@timer
def get_from_prover(endpoint: str) -> requests.Response:
    return requests.get(f"{PROVER_URL}{endpoint}").json()

def run_protocol():
    # 1) Ask the verifier to init a new session: returns {session_id, n, master_seed}
    # init_resp = requests.post(f"{VERIFIER_URL}/init")
    params = {"n": 20000}
    init_resp = post_signed_json("/init", data=params)
    init_data = init_resp.json()
    session_id = init_data["session_id"]
    n = init_data["n"]
    master_seed_hex = init_data["master_seed"]
    print("Session ID:", session_id, "   n =", n)

    # 2) Tell the prover to set A,B using n and master_seed
    setAB_data = post_to_prover(
        "/setAB",
        json={"n": n, "seed": master_seed_hex}
    )
    # check success
    if "status" not in setAB_data or setAB_data["status"] != "ok":
        print("Error setting A,B. Exiting.")
        return
    getCommitment_data = get_from_prover("/getCommitment")
    commitment_root_hex = getCommitment_data["commitment_root"]

    # 3) Pass the prover's commitment_root to the verifier -> get a challenge vector r
    challenge_resp = post_signed_json(
        "/commitment",
        data={
            "session_id": session_id,
            "commitment_root": commitment_root_hex
        }
    )
    challenge_data = challenge_resp.json()
    challenge_vector = challenge_data["challenge_vector"]

    # 4) Send challenge vector to the prover to compute C*r
    computeCR_data = post_to_prover(
        "/computeCR",
        json={"r": challenge_vector}
    )
    Cr = computeCR_data["Cr"]

    # 5) Ask the verifier which rows it wants to check, passing Cr for the Freivalds test
    #    Assume we encode Cr as a comma-separated string for the verifier’s rowchallenge endpoint
    # Cr_str = ",".join(str(x) for x in Cr)
    rowchallenge_resp = post_signed_json(
        "/row_challenge",
        data={
            "session_id": session_id,
            "Cr": Cr
        }
    )
    rowchallenge_data = rowchallenge_resp.json()
    freivalds_ok = rowchallenge_data["freivalds_ok"]
    if not freivalds_ok:
        print("Freivalds check failed. Exiting.")
        return

    # After receiving the spot_rows from /rowchallenge
    spot_rows = rowchallenge_data["spot_rows"]
    print("Freivalds check passed. Spot-checking rows:", spot_rows)

    # 6) Ask the prover for proofs of each row in one call: /getRowProofs
    #    Instead of calling /getRowProof for each row_idx, we can pass an array of row_idxs.
    rowproofs_data = post_to_prover(
        "/getRowProofs",
        json={"row_idxs": spot_rows}
    )

    # Suppose the prover's response is of the form:
    # { "rows": [
    #       { "row_idx": 5,  "row_data": [...], "merkle_path": [...] },
    #       { "row_idx": 11, "row_data": [...], "merkle_path": [...] },
    #       ...
    #   ]
    # }

    # 7) Send all these row proofs to the verifier in one go: /multi_row_check
    payload = {
        "session_id": session_id,
        "rows": rowproofs_data["rows"]
    }
    rowcheck_resp = post_signed_json("/multi_row_check", data=payload)
    rowcheck_result = rowcheck_resp.json()
    # Example response:
    # {
    #   "all_passed": true,
    #   "results": [
    #       { "row_idx": 5,  "pass": true },
    #       { "row_idx": 11, "pass": true }
    #   ]
    # }

    if not rowcheck_result["all_passed"]:
        print("One or more row checks failed:")
        for r in rowcheck_result["results"]:
            if not r["pass"]:
                print(f"Row check failed on row {r['row_idx']}")
        return

    print("All spot-checks succeeded. Verification complete.")

if __name__ == "__main__":
    run_protocol()
