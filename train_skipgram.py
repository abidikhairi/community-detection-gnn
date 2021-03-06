import argparse
from project.utils.utils import adjacency_to_nxg
from project.models.deepwalk import DeepWalk
from project.models.n2v import Node2VecTrainer

def main(args):
    adj_file = args.adj
    model = args.model
    num_walks = args.num_walks
    walk_length = args.walk_length
    vector_size = args.vector_size
    epochs = args.epochs
    window_size = args.window_size
    negatives = args.negatives
    output = args.output

    graph = adjacency_to_nxg(adj_file)

    if model == 'deepwalk':
        model = DeepWalk(graph, num_walks, walk_length, vector_size, epochs, window_size, negatives)
    elif model == 'node2vec':
        model = Node2VecTrainer(graph, num_walks, walk_length, vector_size, epochs, window_size, negatives)
    else:
        raise ValueError(f'Unknown model {model}, available models are deepwalk & node2vec')

    model.train()
    model.save_embeddings(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--adj', type=str, required=True, help='Path to adjacency matrix')
    parser.add_argument('--model', type=str, required=True, choices=['deepwalk', 'node2vec'], help='Model to train (Deepwalk or Node2Vec)')
    parser.add_argument('--num-walks', type=int, default=10, help='Number of random walks to use. Defaults: 10')
    parser.add_argument('--walk-length', type=int, default=10, help='Length of each random walk. Defaults: 10')
    parser.add_argument('--p', type=float, default=1.0, help='Return probability for node2vec. Defaults: 1.0')
    parser.add_argument('--q', type=float, default=1.0, help='Exploring probability for node2vec. Defaults: 1.0')
    parser.add_argument('--vector-size', type=int, default=128, help='Dimension of the embedding vectors. Defaults: 128')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train. Defaults: 10')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for skipgram. Defaults: 10')
    parser.add_argument('--negatives', type=int, default=5, help='Number of negative examples to use. Defaults: 5')
    parser.add_argument('--output', type=str, required=True, help='Path to output embeddings')

    args = parser.parse_args()
    main(args)
