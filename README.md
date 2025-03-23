# GPU Matrix Multiplication Verification Protocol

This protocol verifies a large matrix multiplication $C = A \times B$ performed by an untrusted GPU worker, while keeping the verifier’s computation much cheaper than $O(n^3)$.

## Core Idea

1. **Freivalds’ Algorithm**  
   - The verifier keeps matrices $A$ and $B$ (both $n \times n$).  
   - The worker computes $C = A \times B$.  
   - The verifier chooses a random challenge vector $\mathbf{r}$ (size $n$), which is kept secret until after $C$ is computed.  
   - The worker sends back $C \mathbf{r}$.  
   - The verifier computes $A \bigl(B \mathbf{r}\bigr)$ (only $O(n^2)$ complexity) and checks it against $C \mathbf{r}$.  
   - If they match (within numerical tolerance), $C$ is **very likely** correct.

2. **Merkle‐based Commitment**  
   - The worker **commits** to $C$ by building a Merkle tree over all rows of $C$ and sending the **Merkle root** as a binding commitment.  
   - Each row $C[i, :]$ is hashed to form a leaf of the Merkle tree.  
   - The tree is built level by level, and the final root acts as a single “fingerprint” of $C$.

3. **Spot Checks**  
   - After seeing the Merkle root, the verifier reveals $\mathbf{r}$.  
   - The worker returns $C \mathbf{r}$ along with **selected rows** of $C$ (for instance, randomly chosen), plus Merkle authentication paths that prove those rows are consistent with the committed root.  
   - The verifier recomputes those rows locally by doing $A[i,:] \times B$, an $O(n)$ operation per row, and checks them against the opened rows from $C$.  
   - The Merkle paths ensure the worker cannot produce inconsistent rows without invalidating the commitment root.

## Steps in Detail

1. **Verifier Picks (n, seed) Which Generate $A, B$:**  
   - Matrices $A, B \in \mathbb{R}^{n \times n}$ (or $\mathbb{F}_p$ in a finite field variant).

2. **Worker Computes $C$:**  
   - Receives (n, seed) and recreates $A, B$.  
   - Computes $C = A \times B$ (cost $O(n^3)$ GPU work).  
   - Builds the **Merkle tree** of row hashes $\{H(C[0,:]), \dots, H(C[n-1,:])\}$.  
   - Sends the **Merkle root** to the verifier.

3. **Verifier Sends Random Vector $\mathbf{r}$:**  
   - Kept secret until after the Merkle root is received.

4. **Worker Responds:**  
   - Sends $C \mathbf{r}$.  
   - Opens selected rows: for each chosen row $i$, sends row data and the Merkle authentication path.

5. **Verifier Verifies:**  
   - **Freivalds Check:** Compares $C \mathbf{r}$ to $A (B \mathbf{r})$ in $O(n^2)$ time.  
   - **Row Spot‐Check:** For each opened row $i$, verifies the Merkle path matches the root, then checks the row’s correctness by computing $A[i,:] \times B$.  

If all checks pass, the verifier concludes $C$ is correct with high probability, without doing a full $O(n^3)$ recomputation.

## Usage

1. Generate a key pair:
```python
from eth_account import Account

# Generate a new random account
account = Account.create()
private_key_hex = account.key.hex()

# Print it to set as environment variable
print(f"Generated private key: {private_key_hex}")
print(f"Generated address: {account.address}")
```

2. Create 3 terminals sessions for the prover, verifier and coordinator.

3. Set the private key as an environment variable in the coordinator terminal:
```bash
export PRIVATE_KEY="0x..."
```

4. Set the authorized address in the verifier service:
```bash
export AUTHORIZED_ADDRESS="0x..."
```

5. Run the 3 scripts:
```bash
python prover.py
python verifier_service.py
python coordinator.py
```
