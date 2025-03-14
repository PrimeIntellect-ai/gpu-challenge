import requests

VERIFIER_URL = "http://localhost:14141"
PROVER_URL   = "http://localhost:12121"

def run_protocol():
    # 1) Ask the verifier to init a new session: returns {session_id, n, master_seed}
    init_resp = requests.post(f"{VERIFIER_URL}/init")
    init_data = init_resp.json()
    session_id = init_data["session_id"]
    n = init_data["n"]
    master_seed_hex = init_data["master_seed"]
    print("Session ID:", session_id, "   n =", n)

    # 2) Tell the prover to set A,B using n and master_seed
    setAB_resp = requests.post(
        f"{PROVER_URL}/setAB",
        json={"n": n, "seed": master_seed_hex}
    )
    setAB_data = setAB_resp.json()
    # check success
    if "status" not in setAB_data or setAB_data["status"] != "ok":
        print("Error setting A,B. Exiting.")
        return
    getCommitment_resp = requests.get(f"{PROVER_URL}/getCommitment")
    getCommitment_data = getCommitment_resp.json()
    commitment_root_hex = getCommitment_data["commitment_root"]

    # 3) Pass the prover's commitment_root to the verifier -> get a challenge vector r
    commitment_resp = requests.post(
        f"{VERIFIER_URL}/commitment",
        data={
            "session_id": session_id,
            "commitment_root": commitment_root_hex
        }
    )
    challenge_data = commitment_resp.json()
    challenge_vector = challenge_data["challenge_vector"]

    # 4) Send challenge vector to the prover to compute C*r
    computeCR_resp = requests.post(
        f"{PROVER_URL}/computeCR",
        json={"r": challenge_vector}
    )
    computeCR_data = computeCR_resp.json()
    Cr = computeCR_data["Cr"]

    # 5) Ask the verifier which rows it wants to check, passing Cr for the Freivalds test
    #    Assume we encode Cr as a comma-separated string for the verifier’s rowchallenge endpoint
    Cr_str = ",".join(str(x) for x in Cr)
    rowchallenge_resp = requests.post(
        f"{VERIFIER_URL}/row_challenge",
        data={
            "session_id": session_id,
            "Cr": Cr_str
        }
    )
    rowchallenge_data = rowchallenge_resp.json()
    freivalds_ok = rowchallenge_data["freivalds_ok"]
    if not freivalds_ok:
        print("Freivalds check failed. Exiting.")
        return

    spot_rows = rowchallenge_data["spot_rows"]
    print("Freivalds check passed. Spot-checking rows:", spot_rows)

    # After receiving the spot_rows from /rowchallenge
    spot_rows = rowchallenge_data["spot_rows"]
    print("Freivalds check passed. Spot-checking rows:", spot_rows)

    # 6) Ask the prover for proofs of each row in one call: /getRowProofs
    #    Instead of calling /getRowProof for each row_idx, we can pass an array of row_idxs.
    rowproofs_resp = requests.post(
        f"{PROVER_URL}/getRowProofs",
        json={"row_idxs": spot_rows}
    )
    rowproofs_data = rowproofs_resp.json()

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
    rowcheck_resp = requests.post(
        f"{VERIFIER_URL}/multi_row_check",
        json=payload
    )
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
