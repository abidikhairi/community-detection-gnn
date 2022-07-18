import argparse
import torch as th


def main(args):
    mx = th.load(args.input)

    if args.representation == 'lap':
        D = th.diag(th.sum(mx, dim=0))
        mx = D - mx

    U, _, _ = th.svd_lowrank(mx, args.rank)

    th.save(U, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='The graph adjacency matrix')
    parser.add_argument('--representation', required=True, choices=['adj', 'lap'], help='Representation matrix to use')
    parser.add_argument('--rank', default=128, help='Embedding vector size. Defaults: 128')
    parser.add_argument('--output', type=str, required=True, help='Path to output embeddings')

    args = parser.parse_args()
    main(args)
