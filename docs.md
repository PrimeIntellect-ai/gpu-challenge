# API Documentation

- [API Documentation](#api-documentation)
   * [Verifier](#verifier)
      + [Endpoints](#endpoints)
         - [1. `POST /init`](#1-post-init)
            * [Request Body](#request-body)
            * [Response Body](#response-body)
         - [2. `POST /commitment`](#2-post-commitment)
            * [Request Body](#request-body-1)
            * [Response Body](#response-body-1)
         - [3. `POST /row_challenge`](#3-post-row_challenge)
            * [Request Body](#request-body-2)
            * [Response Body](#response-body-2)
         - [4. `POST /multi_row_check`](#4-post-multi_row_check)
            * [Request Body](#request-body-3)
            * [Response Body](#response-body-3)
         - [5. `POST /clear`](#5-clear)
            * [Request Body](#request-body-4)
            * [Response Body](#response-body-4)
   * [Prover](#prover)
      + [Endpoints](#endpoints-1)
         - [1. `POST /setAB`](#1-post-setab)
            * [Request Body](#request-body-5)
            * [Response Body](#response-body-5)
         - [2. `GET /getCommitment`](#2-get-getcommitment)
            * [Response Body](#response-body-6)
         - [3. `POST /computeCR`](#3-post-computecr)
            * [Request Body](#request-body-6)
            * [Response Body](#response-body-7)
         - [4. `POST /getRowProofs`](#4-post-getrowproofs)
            * [Request Body](#request-body-7)
            * [Response Body](#response-body-8)

## Verifier

This Tornado-based HTTP API verifies matrix multiplication proofs using Freivalds' algorithm and spot-checking rows. Float vectors and matrices are transmitted in Base64 form (raw bytes of the underlying float array).

Requests must be signed with a private key that corresponds to the `ALLOWED_ADDRESS` environment variable.

### Endpoints

---

#### 1. `POST /init`

Creates a new session, generates a pair of deterministic matrices $\(A\)$ and $\(B\)$, and returns:
- A UUID-based `session_id`
- The square matrix dimension `n`
- A 16-byte `master_seed` (hex-encoded) for deterministic row generation.

##### Request Body
```json
{
  "n": 16384     // optional; default is 16384
}
```

##### Response Body
```json
{
  "session_id": "some-uuid",
  "n": 16384,
  "master_seed": "abcd1234..."   // hex-encoded
}
```

---

#### 2. `POST /commitment`

After instructing the prover to set $\(A\)$ and $\(B\)$, submit the Merkle root of the claimed product matrix $\(C\)$. The API returns a random challenge vector $\(r\)$ for Freivalds’ check.

##### Request Body
```json
{
  "session_id": "some-uuid",
  "commitment_root": "abcdef..."  // hex-encoded
}
```

##### Response Body
```json
{
  "challenge_vector": "base64-of-float-array"
}
```
The `challenge_vector` is a Base64-encoded float array, serialized in the same format used internally (e.g. 32-bit floats).

---

#### 3. `POST /row_challenge`

Sends the vector $\(C \cdot r\)$ so the API can perform Freivalds’ check. If the check passes, a set of row indices is returned for spot-checking.

##### Request Body
```json
{
  "session_id": "some-uuid",
  "Cr": "base64-of-float-array"
}
```
`Cr` is the Base64-encoded result of $\(C \cdot r\)$.

##### Response Body
```json
{
  "freivalds_ok": true,
  "spot_rows": [12, 999, ...]
}
```
If `freivalds_ok` is false, the spot rows list is empty.

---

#### 4. `POST /multi_row_check`

Performs a final spot-check on multiple rows. Each row’s content is Merkle-verified and numerically compared.

##### Request Body
```json
{
  "session_id": "some-uuid",
  "rows": [
    {
      "row_idx": 5,
      "row_data": "base64-of-float-array",
      "merkle_path": ["abcd...", "1234...", ...] // list of hex-encoded siblings
    },
    ...
  ]
}
```
- `row_data` is the Base64-encoded row of $\(C\)$ at `row_idx`.

##### Response Body
```json
{
  "all_passed": true,
  "results": [
    {
      "row_idx": 5,
      "pass": true
    },
    ...
  ]
}
```
If `all_passed` is false, at least one row failed verification. The session is freed after this call.

---

#### 5. `POST /clear`

Manually deletes a session prior to its completion or timeout, if the completion is no longer needed.

##### Request Body
```json
{
  "session_id": "some-uuid",
}
```

##### Response Body
```json
{
  "status": "ok"
}
```

If the session does not exist, this call will return a `500` error.

---

## Prover

This Tornado-based HTTP API manages a prover’s side of matrix multiplication verification. It deterministically constructs matrices $\(A\)$ and $\(B\)$ from a seed, computes $\(C = A \times B\)$, and constructs a Merkle tree over the rows of $\(C\)$. Floating-point vectors and rows are serialized in Base64 form to avoid numeric truncation.

### Endpoints

---

#### 1. `POST /setAB`

Generates matrices $\(A\)$ and $\(B\)$ of size $\(\text{n} \times \text{n}\)$ from a seed, computes $\(C = A \times B\)$, and constructs a Merkle tree over $\(C\)$.

##### Request Body
```json
{
  "n": 16384,
  "seed": "abcd1234..." // hex-encoded 16-byte seed
}
```

##### Response Body
```json
{
  "status": "ok"
}
```
On success, the product $\(C\)$ is held in memory, and the Merkle tree and its root are computed.

---

#### 2. `GET /getCommitment`

Returns the Merkle root of $\(C\)$. This root can be used for verifying the integrity of row proofs later.

##### Response Body
```json
{
  "commitment_root": "abcdef..."  // hex-encoded
}
```

---

#### 3. `POST /computeCR`

Computes $\(C \cdot r\)$. The challenge vector `r` is provided in Base64 form (raw bytes of the underlying float array).

##### Request Body
```json
{
  "r": "base64-of-float-array"
}
```

##### Response Body
```json
{
  "Cr": "base64-of-float-array"
}
```
The result is a Base64-encoded float array of length $\(n\)$.

---

#### 4. `POST /getRowProofs`

Given a list of row indices, returns the corresponding rows of $\(C\)$, along with their Merkle proof paths.

##### Request Body
```json
{
  "row_idxs": [12, 999, ...]
}
```

##### Response Body
```json
{
  "rows": [
    {
      "row_idx": 12,
      "row_data": "base64-of-float-array",        // The entire row's bytes
      "merkle_path": ["abcd...", "1234...", ...]  // hex-encoded list of siblings
    },
    ...
  ]
}
```
The `row_data` field is Base64-encoded raw bytes of the float array for row `row_idx`. The `merkle_path` is a list of hexadecimal-encoded sibling hashes proving that row’s membership under the `commitment_root`.