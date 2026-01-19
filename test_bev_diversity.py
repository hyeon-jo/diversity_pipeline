"""Simple test script for BEV diversity pipeline."""

import numpy as np
from bev_diversity.metrics.vendi_family import VendiScoreFamily
from bev_diversity.metrics.similarity import SimilarityMatrix


def test_vendi_basic():
    """Test basic Vendi Score computation."""
    print("Testing basic Vendi Score computation...")

    # Create synthetic embeddings
    np.random.seed(42)
    embeddings = np.random.randn(100, 512).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Compute Vendi scores
    vendi = VendiScoreFamily(kernel="cosine")
    scores = vendi.compute_all(embeddings, [0.1, 0.5, 1.0, 2.0], include_infinity=True)

    print(f"  Number of items: {len(embeddings)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print("\n  Vendi Scores:")
    for key, value in sorted(scores.items()):
        print(f"    {key}: {value:.2f}")

    # Check monotonicity: VS_inf <= VS_2 <= VS_1 <= VS_0.5 <= VS_0.1
    assert scores["VS_q=inf"] <= scores["VS_q=2.0"], "Monotonicity violated: inf > 2.0"
    assert scores["VS_q=2.0"] <= scores["VS_q=1.0"], "Monotonicity violated: 2.0 > 1.0"
    assert scores["VS_q=1.0"] <= scores["VS_q=0.5"], "Monotonicity violated: 1.0 > 0.5"
    assert scores["VS_q=0.5"] <= scores["VS_q=0.1"], "Monotonicity violated: 0.5 > 0.1"

    print("\n  ✓ Monotonicity check passed")
    print()


def test_vendi_identical_items():
    """Test with all identical items - should give score close to 1."""
    print("Testing with identical items...")

    # Create identical embeddings
    n = 50
    embeddings = np.ones((n, 128), dtype=np.float32)

    vendi = VendiScoreFamily(kernel="cosine")
    scores = vendi.compute_all(embeddings, [1.0])

    vs = scores["VS_q=1.0"]
    print(f"  VS (q=1.0) with {n} identical items: {vs:.4f}")
    print(f"  Expected: ~1.0 (minimum diversity)")

    assert vs < 2.0, f"Expected VS close to 1, got {vs}"
    print("  ✓ Test passed")
    print()


def test_vendi_orthogonal_items():
    """Test with orthogonal items - should give score close to n."""
    print("Testing with perfectly diverse items...")

    # Create orthogonal embeddings (identity matrix)
    n = 20
    embeddings = np.eye(n, dtype=np.float32)

    vendi = VendiScoreFamily(kernel="linear")
    scores = vendi.compute_all(embeddings, [1.0])

    vs = scores["VS_q=1.0"]
    print(f"  VS (q=1.0) with {n} orthogonal items: {vs:.4f}")
    print(f"  Expected: ~{n} (maximum diversity)")

    # Should be close to n (within 10% tolerance)
    assert abs(vs - n) / n < 0.1, f"Expected VS close to {n}, got {vs}"
    print("  ✓ Test passed")
    print()


def test_similarity_kernels():
    """Test different similarity kernels."""
    print("Testing different similarity kernels...")

    embeddings = np.random.randn(50, 128).astype(np.float32)

    kernels = ["cosine", "rbf", "linear"]
    for kernel in kernels:
        vendi = VendiScoreFamily(kernel=kernel)
        scores = vendi.compute_all(embeddings, [1.0])
        vs = scores["VS_q=1.0"]
        print(f"  {kernel:8s} kernel: VS = {vs:.2f}")

    print("  ✓ All kernels work")
    print()


def test_q_values_sensitivity():
    """Test sensitivity of different q values."""
    print("Testing q value sensitivity...")

    # Create imbalanced dataset: many similar items + few unique items
    np.random.seed(42)

    # 80 very similar items (cluster 1)
    cluster1 = 10 * np.ones((80, 64)) + 0.1 * np.random.randn(80, 64)

    # 15 similar items (cluster 2)
    cluster2 = -5 * np.ones((15, 64)) + 0.1 * np.random.randn(15, 64)

    # 5 unique items (outliers)
    outliers = np.random.randn(5, 64) * 5

    embeddings = np.vstack([cluster1, cluster2, outliers]).astype(np.float32)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    vendi = VendiScoreFamily(kernel="cosine")
    scores = vendi.compute_all(embeddings, [0.1, 0.5, 1.0, 2.0], include_infinity=True)

    print("  Dataset: 80 similar + 15 similar + 5 unique = 100 items")
    print("\n  Vendi Scores:")
    for key, value in sorted(scores.items(), key=lambda x: (x[0] != "VS_q=inf", x[0])):
        q_str = key.split("=")[1]
        print(f"    VS (q={q_str:>4}): {value:>6.2f}")

    # Small q should be higher (sensitive to outliers)
    assert scores["VS_q=0.1"] > scores["VS_q=2.0"], "Expected VS_0.1 > VS_2.0 for imbalanced data"
    print("\n  ✓ Sensitivity test passed")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("BEV Diversity Pipeline - Unit Tests")
    print("=" * 80)
    print()

    try:
        test_vendi_basic()
        test_vendi_identical_items()
        test_vendi_orthogonal_items()
        test_similarity_kernels()
        test_q_values_sensitivity()

        print("=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise

    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
