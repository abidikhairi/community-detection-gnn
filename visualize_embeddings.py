import os
import argparse
import torch as th
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def main(args):
    out_dir = args.out_dir
    method = args.method
    dataset = args.dataset 
    
    filename = os.path.join(out_dir, f'{dataset}-{method}.png')

    features = th.load(args.emb_file).numpy()
    colors = th.load(args.labels).numpy()

    transformed = TSNE(n_components=2).fit_transform(features)

    plt.figure(figsize=(12, 8))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=colors, cmap='Dark2')
    plt.savefig(filename, transparent=True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--emb-file', required=True, help='Path to embedding file.')
    parser.add_argument('--labels', required=True, help='Path to labels file')
    parser.add_argument('--out-dir', required=True, help='Output dir')
    parser.add_argument('--method', required=True, help='Embedding method used to generate feature matrix')
    parser.add_argument('--dataset', required=True, help='Dataset name')

    args = parser.parse_args()
    main(args)
