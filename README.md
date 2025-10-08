# ABIT-Clustering: Adaptive Binary Iterative Threshold Clustering

## Overview

ABIT-Clustering is a Python library implementing an Adaptive Binary Iterative Threshold (ABIT) clustering algorithm. It is designed for hierarchical clustering of sequential data, such as embeddings from tokenized text, in streaming or online scenarios. Unlike traditional clustering methods that require all data upfront, ABIT supports incremental updates (via `partial_fit`) and can enforce memory bounds by limiting total tokens processed.

This library is particularly useful for applications in natural language processing (NLP), where you might need to group similar sentences or tokens dynamically, such as compressing context for large language models (LLMs) or detecting topic shifts in real-time text streams.

### Why This Matters: A Simple Example for Everyone

Imagine you're chatting with an AI like ChatGPT about a long story or a complex topic. The AI has to remember the entire conversation to respond sensibly, but conversations can get really long—thousands of words! Storing everything uses up memory and slows things down. What if the AI could automatically group similar parts of the conversation into "clusters"? For example:

- Early messages about "planning a vacation" get bundled together.
- Later ones about "booking flights" form another group.
- Unrelated tangents, like "what's the weather like," stay separate.

This clustering makes the AI smarter and faster: it only recalls the most relevant groups, ignoring the fluff. In everyday terms, it's like organizing your messy desk drawers—everything's still there, but neatly sorted so you find what you need quickly. ABIT does this automatically for data streams, helping build more efficient AI tools that save time, energy, and computing power. For developers, it's a tool to make apps handle big data without crashing; for users, it means smoother, more responsive tech in apps like search engines or virtual assistants.

## Abstract

Adaptive Binary Iterative Threshold Clustering (ABIT) is a novel hierarchical clustering algorithm designed for streaming data, particularly suited for processing sequential embeddings such as those from tokenized text. Traditional clustering methods often struggle with incremental updates and bounded memory constraints, leading to inefficiencies in real-time applications like natural language processing and context management in AI systems. ABIT addresses these challenges by employing a bottom-up merging strategy that adaptively determines splitting thresholds through binary search, optimizing for user-defined token count constraints (e.g., minimum and maximum split sizes) while enforcing minimum cluster sizes.

The algorithm computes rolling cosine similarity scores over a sliding window of embeddings, iteratively adjusting the threshold to balance cluster granularity and size tolerances. It supports partial fitting for online learning, enabling seamless integration with streaming inputs, and includes an optional maximum token limit to maintain bounded memory by pruning oldest leaves from the resulting binary tree structure. Implemented in Python with NumPy for efficiency, ABIT is tested extensively for edge cases, consistency between batch and incremental modes, and parameter robustness using pytest.

This project demonstrates ABIT's effectiveness on sentence transformer embeddings, producing interpretable tree hierarchies visualized with anytree. Potential applications include dynamic context compression in large language models, anomaly detection in sequences, and scalable data grouping in resource-constrained environments. Future work may explore integration with advanced embedding models and performance benchmarks against established hierarchical clustering techniques.

## Installation

To install ABIT-Clustering, you can use pip:

```bash
pip install abit-clustering
```

Alternatively, clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/abit-clustering.git
cd abit-clustering
pip install -r requirements.txt
```

Requirements:
- numpy
- pytest (for testing)
- sentence_transformers (for examples)
- anytree (for tree visualization)

## Usage

### Basic Example

Here's how to use ABIT in a simple script. This example clusters sentence embeddings from a text using SentenceTransformers.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from anytree import RenderTree
from abit_clustering import ABITClustering  # Assuming installed or from src

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample text
text = """
Artificial intelligence is reshaping the world. Machine learning is key to AI.
Natural language processing helps machines understand text.
"""

# Tokenize and encode (simplified for sentences)
sentences = [s.strip() for s in text.split('.') if s.strip()]
embeddings = model.encode(sentences)
token_counts = np.array([len(model.tokenizer.tokenize(s)) for s in sentences])  # Approximate token counts

# Initialize and fit
clustering = ABITClustering(min_cluster_size=1)  # Relaxed for small example
clustering.fit(embeddings, token_counts)

# Visualize the tree
def create_anytree(cluster_tree, sentences):
    def build_tree(node, parent=None):
        tree_node = RenderTree.Node(f"Cluster {node.label}", parent=parent)
        if not node.children:
            tree_node.name = sentences[node.label]
        for child in node.children:
            build_tree(child, tree_node)
        return tree_node
    return build_tree(cluster_tree)

root = create_anytree(clustering.tree_, sentences)
for pre, _, node in RenderTree(root):
    print(f"{pre}{node.name}")
"""
Cluster 0
├── Cluster 0
│   ├── Artificial intelligence is reshaping the world
│   └── Machine learning is key to AI
└── Natural language processing helps machines understand text
"""
```

This will output a hierarchical tree showing how sentences are clustered based on similarity.

### Streaming Example

For streaming data:

```python
clustering = ABITClustering(max_tokens=100)  # Bound memory

# Process in batches
for batch_embeddings, batch_counts in data_stream:
    clustering.partial_fit(batch_embeddings, batch_counts)
```

## Algorithm Explanation

ABIT builds a binary tree hierarchy by iteratively merging clusters based on adaptive thresholds. Here's a breakdown of each component, including design decisions.

### 1. TreeNode Dataclass

- **Purpose**: Represents nodes in the hierarchical tree. Each node stores a label, mean embedding, total tokens, total samples, and children.
- **Key Fields**:
  - `label`: Identifier (often the index of the first sample).
  - `mean_encoding`: Weighted average of child embeddings for similarity computations.
  - `total_tokens` and `total_samples`: Track size for constraints.
- **Decisions**: Using a dataclass for simplicity and efficiency in Python 3.7+. The tree is binary-ish but allows variable children during merging (though typically binary due to splitting). This structure enables easy traversal and visualization, chosen over libraries like scikit-learn's agglomerative clustering for custom streaming support.

### 2. _find_split_indices

- **Purpose**: Identifies split points in a sequence of similarity scores where scores drop below a threshold.
- **How it Works**: Iterates over scores, collecting indices where `score < threshold`.
- **Decisions**: Simple and efficient O(n) operation. Used to define cluster boundaries. We chose cosine similarity thresholds because they work well with embeddings (normalized vectors), and splitting on low similarity ensures coherent groups. This is inspired by text segmentation techniques but adapted for embeddings.

### 3. ABITClustering Class

#### Initialization (__init__)

- **Parameters**:
  - `threshold_adjustment`: Step size for binary search (default 0.01).
  - `window_size`: For rolling similarity (default 3).
  - `min_split_tokens`, `max_split_tokens`, `split_tokens_tolerance`: Control split sizes.
  - `min_cluster_size`: Ensures no tiny clusters (default 3).
  - `max_tokens`: Optional memory bound.
- **Decisions**: Defaults balance granularity and performance. Window size of 3 captures local context without excessive computation. Token-based constraints prioritize NLP use cases (e.g., LLM context windows), where token count matters more than sample count. Tolerance allows flexibility to avoid infinite loops in threshold search.

#### fit

- **Purpose**: Full batch fitting; resets state and calls `partial_fit`.
- **Decisions**: Provides a familiar scikit-learn-like interface for non-streaming use, while delegating to `partial_fit` for consistency.

#### partial_fit

- **Purpose**: Incremental fitting for new data batches.
- **How it Works**:
  1. Add new samples as leaf clusters.
  2. While >1 clusters: Compute similarities, find optimal threshold, split/merge into parents.
  3. If no merges, create root.
  4. Prune oldest leaves if exceeding `max_tokens`.
- **Decisions**: Bottom-up merging (agglomerative style) suits streaming, as it builds hierarchy incrementally. We use a loop to repeatedly merge until a single root, ensuring a complete tree. Pruning oldest (FIFO) supports time-series data where recent info is more relevant (e.g., chat histories). This avoids recomputing everything, unlike batch methods, for O(n log n) amortized time.

#### _remove_oldest_leaf

- **Purpose**: Prunes the oldest leaf to enforce `max_tokens`.
- **How it Works**: Recursively removes leftmost leaf, updates stats, collapses single-child nodes, renumbers labels.
- **Decisions**: Depth-first left removal assumes sequential addition (oldest first). Collapsing prevents degenerate trees. This keeps the tree balanced and memory-bounded, crucial for embedded devices or long-running streams.

#### _rolling_similarity_scores

- **Purpose**: Computes cosine similarities between cumulative context and next embedding.
- **How it Works**: For each position, average prior window embeddings, compute dot product normalized.
- **Decisions**: Rolling window captures local coherence (e.g., sentence topics). Cosine similarity is robust to magnitude. Window averaging reduces noise; size 3 is empirical—larger would smooth too much, smaller misses context.

#### _find_optimal_threshold

- **Purpose**: Binary searches for a threshold yielding splits within token constraints.
- **How it Works**: Start with median ± std dev range, iterate 100 times: Adjust low/high based on min/median split sizes.
- **Decisions**: Binary search is efficient (log range). Uses median/std for initial bounds to focus on data distribution. Iterations cap computation; tolerance prevents over-optimization. This adaptivity ensures clusters aren't too small (uninformative) or large (lose detail), unlike fixed-threshold methods.

## Design Decisions Overall

- **Streaming Focus**: Traditional algos like KMeans or DBSCAN aren't hierarchical or incremental. ABIT fills this gap for NLP streams.
- **Efficiency**: NumPy for vector ops; no heavy deps.
- **Flexibility**: Params allow tuning for different scales (e.g., small min_split for fine-grained clustering).
- **Testing**: Comprehensive pytest suite covers edges, consistency, ensuring reliability.
- **Visualization**: AnyTree integration for interpretability—trees show "why" clusters form.

## Testing

Run tests with:

```bash
pytest
```

Covers initialization, edge cases, partial_fit consistency, etc.

## Contributing

Pull requests welcome! See issues for todos.

## License

MIT License. See LICENSE file.
