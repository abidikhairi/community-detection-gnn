import json
import networkx as nx
import numpy as np
import torch as th


def edges_file_to_adj(edges_file):
    edges = np.loadtxt(edges_file, delimiter=' ', dtype=np.int32)
    edges = th.from_numpy(edges)
    ones = th.ones(edges.shape[0])

    adj = th.sparse_coo_tensor(indices=edges.t(), values=ones, size=(edges.max() + 1, edges.max() + 1)).to_dense()
        
    return adj


def features_file_to_tensor(features_file, num_nodes):
    features = np.loadtxt(features_file, delimiter=' ', )
    idx = features[:, 0].astype(int)
    features = np.delete(features, 0, 1)
    
    feats = np.zeros((num_nodes, features.shape[1]))
    feats[idx] = features
    
    return th.from_numpy(feats)


def adjacency_to_nxg(adj_file):
    adj = th.load(adj_file)
    
    if th.is_tensor(adj):
        adj = adj.numpy()

    graph = nx.from_numpy_array(adj)

    return graph


def load_communities(path):
    with open(path) as stream:
        return json.load(stream)


def normalize_adj(adj: th.Tensor):
    adj.fill_diagonal_(0) # remove self-loops if exists

    deg_invrt_sqrt = th.diag(th.pow(th.sum(adj, dim=0), -0.5))
    adj = adj + th.eye(adj.shape[0])

    return th.matmul(th.matmul(deg_invrt_sqrt, adj), deg_invrt_sqrt)
