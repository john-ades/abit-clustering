from typing import List
from dataclasses import dataclass
import numpy as np

@dataclass
class TreeNode:
    label: int
    mean_encoding: np.ndarray = None
    total_tokens: int = 0
    total_samples: int = 0
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
            min_cluster_size: int = 3,
            max_tokens: int = None
    ):
        self.threshold_adjustment = threshold_adjustment
        self.window_size = window_size
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.min_cluster_size = min_cluster_size
        self.max_tokens = max_tokens if max_tokens is not None else float('inf')
        self.labels_ = None
        self.tree_ = None
        self.current_clusters = None
        self.current_nodes = None
        self.n_samples_ = 0

    def fit(self, X, T):
        """Fit the clustering model."""
        self.current_clusters = None
        self.current_nodes = None
        self.n_samples_ = 0
        self.tree_ = None
        self.labels_ = None
        self.partial_fit(X, T)

    def partial_fit(self, X, T):
        """Partial fit for streaming data."""
        if self.current_clusters is None:
            self.current_clusters = []
            self.current_nodes = []
            self.labels_ = np.array([], dtype=int)
            self.n_samples_ = 0
            self.tree_ = None
        new_n = X.shape[0]
        if new_n == 0:
            return
        new_labels = np.arange(self.n_samples_, self.n_samples_ + new_n)
        self.labels_ = np.append(self.labels_, new_labels)
        new_clusters = [[self.n_samples_ + i] for i in range(new_n)]
        new_tree_nodes = [
            TreeNode(
                label=int(new_labels[i]),
                mean_encoding=X[i],
                total_tokens=T[i],
                total_samples=1
            ) for i in range(new_n)
        ]
        self.current_clusters += new_clusters
        self.current_nodes += new_tree_nodes
        self.n_samples_ += new_n
        clusters = self.current_clusters
        tree_nodes = self.current_nodes
        while len(clusters) > 1:
            cluster_encodings = [node.mean_encoding for node in tree_nodes]
            cluster_token_counts = [node.total_tokens for node in tree_nodes]
            similarities = self._rolling_similarity_scores(cluster_encodings)
            calculated_threshold = self._find_optimal_threshold(cluster_token_counts, similarities)
            split_indices = [0] + _find_split_indices(similarities, calculated_threshold) + [len(clusters)]
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
            if len(split_indices) > 1:
                last_start = split_indices[-2]
                last_end = split_indices[-1]
                last_size = cumulative_token_counts[last_end] - cumulative_token_counts[last_start]
                if last_size < self.min_cluster_size and len(split_indices) > 2:
                    del split_indices[-2]
            parent_cluster_ranges = list(zip(split_indices[:-1], split_indices[1:]))
            new_clusters = []
            new_tree_nodes = []
            merged = False
            for start_idx, end_idx in parent_cluster_ranges:
                if end_idx - start_idx > 1:
                    merged = True
                    parent_cluster = [item for sublist in clusters[start_idx:end_idx] for item in sublist]
                    parent_label = self.labels_[parent_cluster[0]]
                    self.labels_[parent_cluster] = parent_label
                    children = tree_nodes[start_idx:end_idx]
                    total_samples = sum(c.total_samples for c in children)
                    mean_encoding = np.sum([c.mean_encoding * c.total_samples for c in children], axis=0) / total_samples
                    total_tokens = sum(c.total_tokens for c in children)
                    parent_node = TreeNode(
                        label=int(parent_label),
                        mean_encoding=mean_encoding,
                        total_tokens=total_tokens,
                        total_samples=total_samples,
                        children=children
                    )
                    new_clusters.append(parent_cluster)
                    new_tree_nodes.append(parent_node)
                else:
                    new_clusters.append(clusters[start_idx])
                    new_tree_nodes.append(tree_nodes[start_idx])
            if not merged:
                # No new parent clusters identified, create final root cluster
                root_cluster = [item for sublist in new_clusters for item in sublist]
                root_label = self.labels_[root_cluster[0]]
                self.labels_[root_cluster] = root_label
                total_samples = sum(c.total_samples for c in new_tree_nodes)
                mean_encoding = np.sum([c.mean_encoding * c.total_samples for c in new_tree_nodes], axis=0) / total_samples if total_samples > 0 else np.zeros_like(X[0] if new_n > 0 else np.array([]))
                total_tokens = sum(c.total_tokens for c in new_tree_nodes)
                self.tree_ = TreeNode(
                    label=int(root_label),
                    mean_encoding=mean_encoding,
                    total_tokens=total_tokens,
                    total_samples=total_samples,
                    children=new_tree_nodes
                )
                self.current_clusters = [root_cluster]
                self.current_nodes = [self.tree_]
                break
            else:
                clusters = new_clusters
                tree_nodes = new_tree_nodes
                self.current_clusters = clusters
                self.current_nodes = tree_nodes
        if len(self.current_clusters) == 1:
            self.tree_ = self.current_nodes[0]

        # Enforce max_tokens by removing oldest leaves if necessary
        while self.tree_ and self.tree_.total_tokens > self.max_tokens:
            self._remove_oldest_leaf()
            self.n_samples_ -= 1
            self.labels_ = np.delete(self.labels_, 0)

    def _remove_oldest_leaf(self):
        if not self.tree_:
            return
        if not self.tree_.children:
            self.tree_ = None
            self.current_clusters = []
            self.current_nodes = []
            return

        def remove_and_update(node: TreeNode) -> TreeNode:
            if not node.children:
                return None
            node.children[0] = remove_and_update(node.children[0])
            if node.children[0] is None:
                del node.children[0]
            if not node.children:
                return None
            # Update stats
            node.total_samples = sum(c.total_samples for c in node.children)
            node.total_tokens = sum(c.total_tokens for c in node.children)
            node.mean_encoding = np.sum([c.mean_encoding * c.total_samples for c in node.children], axis=0) / node.total_samples
            # Collapse if single child
            if len(node.children) == 1:
                return node.children[0]
            return node

        self.tree_ = remove_and_update(self.tree_)
        self.current_nodes = [self.tree_] if self.tree_ else []

        # Renumber labels in the tree (subtract 1 since removing the smallest index)
        def renumber(node: TreeNode):
            if not node:
                return
            if not node.children:
                node.label -= 1
            else:
                for child in node.children:
                    renumber(child)
                node.label = min(child.label for child in node.children)

        renumber(self.tree_)

        # Update current_clusters: remove the first index and shift the rest down by 1
        if self.current_clusters:
            self.current_clusters[0].pop(0)
            for i in range(len(self.current_clusters[0])):
                self.current_clusters[0][i] -= 1

    def _rolling_similarity_scores(self, encoded_docs: List[np.ndarray]) -> List[float]:
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
            if (min_tokens >= self.min_split_tokens - self.split_tokens_tolerance and
                median_tokens <= self.max_split_tokens + self.split_tokens_tolerance):
                break
            elif min_tokens < self.min_split_tokens:
                high = calculated_threshold - self.threshold_adjustment  # Lower threshold for larger clusters
            else:
                low = calculated_threshold + self.threshold_adjustment  # Higher threshold for smaller clusters
            if abs(high - low) < 1e-5:
                break
        return calculated_threshold