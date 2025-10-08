import pytest
import numpy as np
from numpy.testing import assert_allclose
from clustering import ABITClustering

@pytest.fixture
def clustering():
    return ABITClustering(
        threshold_adjustment=0.01,
        window_size=3,
        min_split_tokens=5,
        max_split_tokens=10,
        split_tokens_tolerance=5,
        min_cluster_size=3
    )

def test_partial_fit_initialization(clustering):
    # Partial fit with empty data
    clustering.partial_fit(np.empty((0, 2)), np.empty(0, dtype=int))
    assert len(clustering.labels_) == 0
    assert clustering.tree_ is None
    assert clustering.n_samples_ == 0
    assert len(clustering.current_clusters) == 0
    assert len(clustering.current_nodes) == 0

def test_partial_fit_single_batch(clustering):
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    T = np.array([1, 1, 1, 1], dtype=int)
    clustering.partial_fit(X, T)
    assert clustering.n_samples_ == 4
    assert len(set(clustering.labels_)) == 1 # Assuming single root cluster as in simple case
    assert clustering.tree_ is not None
    assert hasattr(clustering.tree_, 'label')
    assert hasattr(clustering.tree_, 'children')

def test_partial_fit_multiple_batches(clustering):
    # First batch
    X1 = np.array([[1, 0], [0, 1]])
    T1 = np.array([1, 1], dtype=int)
    clustering.partial_fit(X1, T1)
    assert clustering.n_samples_ == 2
    assert len(clustering.current_clusters) == 1 # Should merge into one if similar, but depending on data
    assert clustering.tree_ is not None
    # Second batch
    X2 = np.array([[1, 1], [0, 0]])
    T2 = np.array([1, 1], dtype=int)
    clustering.partial_fit(X2, T2)
    assert clustering.n_samples_ == 4
    assert len(set(clustering.labels_)) == 1 # Assuming all merge
    assert clustering.tree_ is not None

def test_partial_fit_consistency_with_fit(clustering):
    X = np.ones((6, 2))
    T = np.ones(6, dtype=int)
    # Fit all at once
    clustering_fit = ABITClustering(
        threshold_adjustment=0.01,
        window_size=3,
        min_split_tokens=5,
        max_split_tokens=10,
        split_tokens_tolerance=5,
        min_cluster_size=3
    )
    clustering_fit.fit(X, T)
    labels_fit = clustering_fit.labels_
    tree_fit = clustering_fit.tree_
    # Partial fit in batches
    clustering.partial_fit(X[:3], T[:3])
    clustering.partial_fit(X[3:], T[3:])
    labels_partial = clustering.labels_
    tree_partial = clustering.tree_
    # Check labels consistency (allowing for label offset but same grouping)
    unique_fit = len(set(labels_fit))
    unique_partial = len(set(labels_partial))
    assert unique_fit == unique_partial
    # Check tree totals
    assert tree_fit.total_samples == tree_partial.total_samples
    assert tree_fit.total_tokens == tree_partial.total_tokens
    assert_allclose(tree_fit.mean_encoding, tree_partial.mean_encoding)

def test_partial_fit_single_samples(clustering):
    for i in range(5):
        X_single = np.array([[1, 0]]) if i % 2 == 0 else np.array([[0, 1]])
        T_single = np.array([1], dtype=int)
        clustering.partial_fit(X_single, T_single)
    assert clustering.n_samples_ == 5
    assert clustering.tree_ is not None
    assert len(clustering.current_clusters) == 1 # Assuming merging occurs

def test_partial_fit_empty_after_nonempty(clustering):
    X = np.array([[1, 0]])
    T = np.array([1], dtype=int)
    clustering.partial_fit(X, T)
    assert clustering.n_samples_ == 1
    # Empty batch
    clustering.partial_fit(np.empty((0, 2)), np.empty(0, dtype=int))
    assert clustering.n_samples_ == 1
    assert clustering.tree_.total_samples == 1

def test_partial_fit_tree_structure(clustering):
    X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    T = np.ones(4, dtype=int)
    # Partial fit in two batches
    clustering.partial_fit(X[:2], T[:2])
    assert clustering.tree_ is not None
    clustering.partial_fit(X[2:], T[2:])
    assert clustering.tree_ is not None
    assert clustering.tree_.label == clustering.labels_[0]
    assert clustering.tree_.total_samples == 4
    assert clustering.tree_.total_tokens == 4

def test_partial_fit_with_max_tokens_single_adds():
    clustering = ABITClustering(
        threshold_adjustment=0.01,
        window_size=3,
        min_split_tokens=5,
        max_split_tokens=10,
        split_tokens_tolerance=5,
        min_cluster_size=3,
        max_tokens=3
    )
    means = []
    totals = []
    for i in range(4):
        X = np.array([[float(i+1), 0.0]])
        T = np.array([1], dtype=int)
        clustering.partial_fit(X, T)
        if clustering.tree_:
            means.append(clustering.tree_.mean_encoding[0])
            totals.append(clustering.tree_.total_tokens)
        else:
            means.append(None)
            totals.append(0)
    assert totals == [1, 2, 3, 3]
    assert_allclose(means, [1.0, 1.5, 2.0, 3.0])
    assert clustering.n_samples_ == 3

def test_partial_fit_with_max_tokens_large_batch():
    clustering = ABITClustering(
        threshold_adjustment=0.01,
        window_size=3,
        min_split_tokens=5,
        max_split_tokens=10,
        split_tokens_tolerance=5,
        min_cluster_size=3,
        max_tokens=3
    )
    X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    T = np.ones(5, dtype=int)
    clustering.partial_fit(X, T)
    assert clustering.n_samples_ == 3
    assert clustering.tree_.total_tokens == 3
    assert_allclose(clustering.tree_.mean_encoding, [4.0, 0.0])