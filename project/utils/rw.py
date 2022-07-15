import random
import networkx as nx


class RandomWalker:
    def __init__(self, graph: nx.Graph, walk_length: int, num_walks: int) -> None:
        
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks

    def get_random_walk(self, start) -> list:
        walk = [start]
        while len(walk) < self.walk_length:
            neighbors = self.graph.neighbors(walk[-1])
            if len(neighbors) == 0:
                next = walk[-1]
            else:
                next = random.choice(neighbors)            
            walk.append(next)

        return walk


    def simulate_walks(self) -> list:
        nodes = random.shuffle(list(self.graph.nodes()))
        walks = []

        for node in nodes:
            for _ in range(self.num_walks):
                walks.append(self.get_random_walk(node))
    
        return walks
