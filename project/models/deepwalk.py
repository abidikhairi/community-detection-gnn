import networkx as nx
import torch as th
from ..utils.rw import RandomWalker
from gensim.models import Word2Vec


class DeepWalk():
    def __init__(self, graph: nx.graph, num_walks: int, walk_length: int, vector_size: int, epochs: int = 10, window_size: int = 10, negatives: int = 5) -> None:
        
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.vector_size = vector_size
        self.epochs = epochs
        self.negatives = negatives
        self.window_size = window_size

        self.walker = RandomWalker(graph, walk_length, num_walks)
        self.embedding = None


    def train(self):
        walks = self.walker.simulate_walks()
        model = Word2Vec(walks, vector_size=self.vector_size, window=self.window_size, min_count=1, sg=1, workers=4, epochs=self.epochs, negative=self.negatives)
        import pdb; pdb.set_trace()
        self.embedding = model.wv


    def save_embeddings(self, filename):
        th.save(th.from_numpy(self.embedding), filename)
