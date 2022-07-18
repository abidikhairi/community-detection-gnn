import torch as th


def get_real_and_non_existing_edges(adj):
    non_edges = th.nonzero(adj == 0, as_tuple=False)
    edges = th.nonzero(adj == 1, as_tuple=False)

    return edges, non_edges


def sample_non_existing_edges(edges, k):
    idx = th.randperm(len(edges))[:k]

    return edges[idx]
