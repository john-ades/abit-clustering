import numpy as np
from sentence_transformers import SentenceTransformer
from anytree import Node, RenderTree

from abit_clustering import ABITClustering

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

text = """
Artificial intelligence is reshaping the world as we know it. Machine learning algorithms are becoming increasingly sophisticated, 
capable of analyzing vast amounts of data and making predictions with remarkable accuracy. Natural language processing has made 
significant strides, enabling machines to understand and generate human-like text. Computer vision systems can now recognize and 
interpret visual information with precision rivaling human capabilities. These advancements are driving innovation across industries, 
from healthcare and finance to transportation and entertainment. As AI continues to evolve, it raises important questions about ethics, 
privacy, and the future of work. Researchers and policymakers are grappling with the challenge of ensuring that AI development 
benefits humanity as a whole. The potential of AI is immense, but so too are the responsibilities that come with its development and deployment.
"""

text_1 = """In 2024, AI patents in China and the US numbered more than three-fourths of AI patents worldwide. 
Though China had more AI patents, the US had 35% more patents per AI patent-applicant company than China. 
The study of mechanical or 'formal' reasoning began with philosophers and mathematicians in antiquity. 
The study of logic led directly to Alan Turing's theory of computation, which suggested that a machine, by shuffling symbols as simple as '0' and '1', could simulate any conceivable form of mathematical reasoning. 
This, along with concurrent discoveries in cybernetics, information theory and neurobiology, led researchers to consider the possibility of building an 'electronic brain'. 
They developed several areas of research that would become part of AI, such as McCulloch and Pitts design for 'artificial neurons' in 1943, and Turing's influential 1950 paper 'Computing Machinery and Intelligence', which introduced the Turing test and showed that 'machine intelligence'."""


text_2 = """It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife. 
However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters. 
'My dear Mr. Bennet,' said his lady to him one day, 'have you heard that Netherfield Park is let at last?' 
Mr. Bennet replied that he had not. 'But it is,' returned she; 'for Mrs. Long has just been here, and she told me all about it.' Mr. Bennet made no answer. 'Do not you want to know who has taken it?' cried his wife, impatiently. 'You want to tell me, and I have no objection to hearing it.' This was invitation enough. 'Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place...'."""



if __name__ == '__main__':
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Sample text
    text = text_1

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
        min_cluster_size=8,
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