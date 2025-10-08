import numpy as np
from sentence_transformers import SentenceTransformer
from anytree import Node, RenderTree

from src.clustering import ABITClustering

def create_anytree(cluster_tree, tokens):
    """Convert cluster tree to anytree structure."""
    def build_tree(node, parent=None):
        tree_node = Node(f"Cluster {node.label}", parent=parent)
        if not node.children:
            tree_node.name = tokens[node.label]
        for child in node.children:
            build_tree(child, tree_node)
        return tree_node
    return build_tree(cluster_tree)

if __name__ == '__main__':
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Sample text
    text = """
    Artificial intelligence is reshaping the world as we know it. 
    Machine learning algorithms are becoming increasingly sophisticated, 
    capable of analyzing vast amounts of data and making predictions with remarkable accuracy.
    """

    # Tokenize the text
    tokens = model.tokenizer.tokenize(text)

    # Encode to get token embeddings (excluding [CLS] and [SEP])
    token_embeddings = model.encode(text, output_value='token_embeddings')[1:-1]

    # Convert to numpy array if necessary
    if not isinstance(token_embeddings, np.ndarray):
        token_embeddings = np.array([emb.cpu().numpy() for emb in token_embeddings])

    token_counts = np.ones(len(tokens), dtype=int)

    # Initialize the clustering model with set min token sizes
    clustering = ABITClustering(
        threshold_adjustment=0.01,
        window_size=3,
        min_split_tokens=5,
        max_split_tokens=10,
        split_tokens_tolerance=5,
        min_cluster_size=3,
        max_tokens=200
    )

    # Fit the model (batch mode for simplicity)
    clustering.fit(token_embeddings, token_counts)

    # Visualize the tree
    print("\n--- Token-Level Clustering Tree ---")
    if clustering.tree_ is None:
        print("No tree (insufficient data for clustering).")
    else:
        root = create_anytree(clustering.tree_, tokens)
        for pre, _, node in RenderTree(root):
            print(f"{pre}{node.name}")

"""--- Token-Level Clustering Tree ---
Cluster 0
├── Cluster 0
│   ├── artificial
│   ├── intelligence
│   └── is
├── Cluster 3
│   ├── Cluster 3
│   │   ├── res
│   │   ├── ##ha
│   │   └── ##ping
│   └── Cluster 6
│       ├── the
│       ├── world
│       ├── as
│       └── we
└── Cluster 10
    ├── Cluster 10
    │   ├── know
    │   ├── it
    │   └── .
    ├── Cluster 13
    │   ├── machine
    │   ├── learning
    │   └── algorithms
    ├── Cluster 16
    │   ├── Cluster 16
    │   │   ├── are
    │   │   ├── becoming
    │   │   ├── increasingly
    │   │   └── sophisticated
    │   └── Cluster 20
    │       ├── ,
    │       ├── capable
    │       └── of
    └── Cluster 23
        ├── Cluster 23
        │   ├── analyzing
        │   ├── vast
        │   └── amounts
        ├── Cluster 26
        │   ├── of
        │   ├── data
        │   └── and
        ├── Cluster 29
        │   ├── making
        │   ├── predictions
        │   └── with
        └── Cluster 32
            ├── remarkable
            ├── accuracy
            └── .
"""