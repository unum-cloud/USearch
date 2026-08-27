from usearch.index import Index, Indexes
import numpy as np

def test_multi_index():
    print("Initiating Stress Protocol...")

    # ---------------------------------------------------------
    # Phase 1: The Void Strike (Boundary Conditions)
    # Proving that an empty Indexes object does not crash the engine.
    # ---------------------------------------------------------
    print("Executing Phase 1: Boundary Conditions...")
    ix_empty = Indexes([])
    assert ix_empty.ndim == 0, f"Void ndim failed. Expected 0, got {ix_empty.ndim}"
    assert ix_empty.dtype is None, "Void dtype failed."
    assert ix_empty.metric == "unknown", f"Void metric failed. Expected 'unknown', got {ix_empty.metric}"

    ndim = 128
    v1 = np.random.rand(ndim).astype(np.float32)
    v2 = np.random.rand(ndim).astype(np.float32)

    ia = Index(ndim=ndim)
    ia.add(100, v1)  # Key 100 holds v1

    ib = Index(ndim=ndim)
    ib.add(200, v2)  # Key 200 holds v2

    ix = Indexes([ia, ib])

    # ---------------------------------------------------------
    # Phase 2: The Recall Strike (Mathematical Verification)
    # Proving the multi-index router searches the correct memory space.
    # ---------------------------------------------------------
    print("Executing Phase 2: Recall...")
    matches = ix.search(v2, count=1)
    assert len(matches) > 0, "Engine returned empty results."
    assert matches[0].key == 200, f"Recall failure. Expected 200, got {matches[0].key}"
    # Distance to itself should be essentially zero
    assert matches[0].distance < 1e-5, f"Distance variance too high: {matches[0].distance}"

    # ---------------------------------------------------------
    # Phase 3: The Batch Strike (Memory Alignment)
    # Proving the wrapper can handle matrices, not just single vectors.
    # ---------------------------------------------------------
    print("Executing Phase 3: Batch Memory Alignment...")
    batch_query = np.vstack([v1, v2]) # Shape: (2, 128)
    
    batch_matches = ix.search(batch_query, count=1)
    assert len(batch_matches) == 2, f"Batch router failed. Expected 2 result sets, got {len(batch_matches)}"
    
    # The first query (v1) should return key 100
    assert batch_matches[0][0].key == 100, f"Batch recall 0 failed. Got {batch_matches[0][0].key}"
    # The second query (v2) should return key 200
    assert batch_matches[1][0].key == 200, f"Batch recall 1 failed. Got {batch_matches[1][0].key}"

    print("All assertions passed. The architecture holds.")

if __name__ == "__main__":
    test_multi_index()