import torch as th
import dgl.data as datasets


def main():
    cora = datasets.CoraGraphDataset(reverse_edge=False)
    graph = cora[0]

    adj = graph.adj().to_dense()
    clusters = graph.ndata['label'].squeeze()
    nfeats = graph.ndata['feat']
    
    th.save(adj, 'data/cora/adj.pt')
    th.save(clusters, 'data/cora/clusters.pt')
    th.save(nfeats, 'data/cora/nfeats.pt')

    citeseer = datasets.CiteseerGraphDataset(reverse_edge=False)
    graph = citeseer[0]

    adj = graph.adj().to_dense()
    clusters = graph.ndata['label'].squeeze()
    nfeats = graph.ndata['feat']

    th.save(adj, 'data/citeseer/adj.pt')
    th.save(clusters, 'data/citeseer/clusters.pt')
    th.save(nfeats, 'data/citeseer/nfeats.pt')


if __name__ == '__main__':
    main()
