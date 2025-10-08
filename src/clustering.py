from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass
class TreeNode:
    label: int
    children: List['TreeNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def _find_split_indices(similarities: List[float], threshold: float) -> List[int]:
    """Find indices where splits should occur based on similarity scores."""
    return [idx + 1 for idx, score in enumerate(similarities) if score < threshold]


class ABITClustering:
    """
    Adaptive Binary Iterative Threshold Clustering
    """

    def __init__(
            self,
            threshold_adjustment: float = 0.01,
            window_size: int = 3,
            min_split_tokens: int = 5,
            max_split_tokens: int = 10,
            split_tokens_tolerance: int = 5,
            min_cluster_size: int = 3
    ):
        self.threshold_adjustment = threshold_adjustment
        self.window_size = window_size
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.tree_ = None


    def fit(self, X, T):
        """Fit the clustering model."""
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)

        clusters = [[i] for i in range(n_samples)]
        tree_nodes = [TreeNode(i) for i in range(n_samples)]

        while len(clusters) > 1:
            cluster_encodings = [np.mean([X[i] for i in cluster], axis=0) for cluster in clusters]
            cluster_token_counts = [sum(T[i] for i in cluster) for cluster in clusters]

            similarities = self._rolling_similarity_scores(cluster_encodings)
            calculated_threshold = self._find_optimal_threshold(cluster_token_counts, similarities)

            split_indices = [0] + _find_split_indices(similarities, calculated_threshold) + [len(clusters)]

            # Enforce min cluster size by removing splits that create small groups
            cumulative_token_counts = np.cumsum([0] + cluster_token_counts)
            i = 1
            while i < len(split_indices) - 1:
                start = split_indices[i - 1]
                end = split_indices[i]
                size = cumulative_token_counts[end] - cumulative_token_counts[start]
                if size < self.min_cluster_size:
                    del split_indices[i]
                else:
                    i += 1
            # Check the last group
            if len(split_indices) > 1:
                last_start = split_indices[-2]
                last_end = split_indices[-1]
                last_size = cumulative_token_counts[last_end] - cumulative_token_counts[last_start]
                if last_size < self.min_cluster_size:
                    del split_indices[-2]  # Merge last with previous

            parent_cluster_ranges = list(zip(split_indices[:-1], split_indices[1:]))

            new_clusters = []
            new_tree_nodes = []
            for start_idx, end_idx in parent_cluster_ranges:
                parent_cluster = [item for sublist in clusters[start_idx:end_idx] for item in sublist]
                new_clusters.append(parent_cluster)
                parent_label = self.labels_[parent_cluster[0]]
                self.labels_[parent_cluster] = parent_label

                parent_tree_node = TreeNode(int(parent_label), tree_nodes[start_idx:end_idx])
                new_tree_nodes.append(parent_tree_node)

            if len(new_clusters) == len(clusters):
                # No new parent clusters identified, create final root cluster
                root_cluster = [item for sublist in new_clusters for item in sublist]
                root_label = self.labels_[root_cluster[0]]
                self.labels_[root_cluster] = root_label
                self.tree_ = TreeNode(root_label, new_tree_nodes)
                break

            clusters = new_clusters
            tree_nodes = new_tree_nodes

        if self.tree_ is None:
            self.tree_ = tree_nodes[0]

    def _rolling_similarity_scores(self, encoded_docs: List[List[float]]) -> List[float]:
        """Calculate rolling similarity scores."""
        encoded_docs = np.array(encoded_docs)
        similarities = []
        for idx in range(1, len(encoded_docs)):
            window_start = max(0, idx - self.window_size)
            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            similarity = np.dot(cumulative_context, encoded_docs[idx]) / (
                    np.linalg.norm(cumulative_context) * np.linalg.norm(encoded_docs[idx]) + 1e-10
            )
            similarities.append(similarity)
        return similarities

    def _find_optimal_threshold(self, token_counts: List[int], similarity_scores: List[float]) -> float:
        """Find the optimal threshold for splitting clusters."""
        cumulative_token_counts = np.cumsum([0] + token_counts)

        median_score = np.median(similarity_scores)
        std_dev = np.std(similarity_scores)

        low = max(0.0, float(median_score - std_dev))
        high = min(1.0, float(median_score + std_dev))

        calculated_threshold = 0.0
        for _ in range(100):  # Max 100 iterations
            calculated_threshold = (low + high) / 2
            split_indices = _find_split_indices(similarity_scores, calculated_threshold)

            split_token_counts = [
                cumulative_token_counts[end] - cumulative_token_counts[start]
                for start, end in zip([0] + split_indices, split_indices + [len(token_counts)])
            ]

            if not split_token_counts:
                high = calculated_threshold - self.threshold_adjustment
                continue

            min_tokens = np.min(split_token_counts)
            median_tokens = np.median(split_token_counts)

            if (min_tokens >= self.min_cluster_size - self.split_tokens_tolerance and
                median_tokens <= self.max_split_tokens + self.split_tokens_tolerance):
                break
            elif min_tokens < self.min_cluster_size:
                high = calculated_threshold - self.threshold_adjustment  # Lower threshold for larger clusters
            else:
                low = calculated_threshold + self.threshold_adjustment  # Higher threshold for smaller clusters

            if abs(high - low) < 1e-5:
                break

        return calculated_threshold
