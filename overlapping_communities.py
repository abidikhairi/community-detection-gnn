import argparse
import json
import torch as th
import torch.optim as optim

from project.models.mlp import Perceptron
from project.models.gnn import GCN
from project.utils.sampling import get_real_and_non_existing_edges, sample_non_existing_edges
from project.utils.utils import load_communities, normalize_adj, get_predicted_communities
from project.utils.onmi import communities_to_onmi, exec_onmi


def main(args):

    adj = th.load(args.adj).float()
    feats = th.load(args.feats).float()
    communities = load_communities(args.communities) 

    # save ground truth labels
    communities_to_onmi(communities, 'tmp/ground_truth.txt')
    
    ncomms = len(communities.keys())
    nfeats = feats.shape[1]

    edges, non_edges = get_real_and_non_existing_edges(adj)

    if args.model == 'mlp':
        model = Perceptron(nfeats, ncomms)
    elif args.model == 'gcn':
        adj = normalize_adj(adj)
        model = GCN(nfeats, args.nhids, ncomms, adj)
    else:
        raise ValueError('unknown model {}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for _ in range(args.epochs):
        optimizer.zero_grad()

        F = model(feats)
        F = th.sigmoid(F)

        src, dst = edges[:, 0], edges[:, 1]
        pos_loss = - th.log(1 - th.exp(-th.diag(th.matmul(F[src], F[dst].t())))).mean()
    
        fake_edges = sample_non_existing_edges(non_edges, len(edges))
        src, dst = fake_edges[:, 0], fake_edges[:, 1]
    
        neg_loss = th.diag(th.matmul(F[src], F[dst].t())).mean()
        loss = pos_loss + neg_loss
    
        loss.backward()
        optimizer.step()
    
    y_hat = get_predicted_communities(F.detach(), 0.4)
    communities_to_onmi(y_hat, 'tmp/y_hat.txt')
    
    result = exec_onmi('tmp/ground_truth.txt', 'tmp/y_hat.txt')

    print(json.dumps(result, sort_keys=False, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--adj', required=True, help='Path to adjacency matrix')
    parser.add_argument('--feats', required=True, help='Path to node features matrix')
    parser.add_argument('--communities', required=True, help='Path to communities dict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs. Defaults: 100')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Defaults: 0.001')
    parser.add_argument('--model', required=True, choices=['mlp', 'gcn'], help='The model to train (mlp or gcn)')
    parser.add_argument('--nhids', type=int, default=16, help='Number of hidden units (only for gcn)')

    args = parser.parse_args()
    main(args)
