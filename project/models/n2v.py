import networkx as nx
import torch as th
from node2vec import Node2Vec


class Node2VecTrainer():


    def __init__(self, graph: nx.graph, num_walks: int, walk_length: int, vector_size: int, epochs: int = 10, window_size: int = 10, negatives: int = 5, p: float = 1.0, q: float = 1.0) -> None:
        
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.vector_size = vector_size
        self.epochs = epochs
        self.negatives = negatives
        self.window_size = window_size

        self.embedding = None


    def train(self):
        model = Node2Vec(self.graph, dimensions=self.vector_size, walk_length=self.walk_length,
            num_walks=self.num_walks, p=self.p, q=self.q
        )

        model.fit(epochs=self.epochs, window=self.window_size, negative=self.negatives)
        keys = model.wv.key_to_index
        embeddings = th.zeros(len(keys), self.vector_size)

        for key, idx in keys.items():
            embeddings[idx] = th.tensor(model.wv[key])

        self.embedding = embeddings


    def save_embeddings(self, filename):
        th.save(self.embedding, filename)
