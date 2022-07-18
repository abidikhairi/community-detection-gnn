import argparse
import torch as th
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score


def main(args):
    method = args.method
    algorithm = args.clustering

    features = th.load(args.input).numpy()
    labels = th.load(args.labels).numpy()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.15)

    num_clusters = max(labels) + 1

    if algorithm == 'kmeans':
        model = KMeans(n_clusters=num_clusters)
    elif algorithm == 'sc':
        model = SpectralClustering(n_clusters=num_clusters)
    else:
        raise ValueError('Algorithm {} not supported')

    model.fit(x_train)

    y_hat = model.predict(x_test) if algorithm == 'kmeans' else model.fit_predict(x_test)
    
    nmi = normalized_mutual_info_score(y_test, y_hat)
    ami = adjusted_mutual_info_score(y_test, y_hat)

    print(f'Embedding Method: {method}')
    print(f'Clustering Algorithm {algorithm}')
    print(f'NMI score: {nmi*100:.2f} %')
    print(f'AMI score: {ami*100:.2f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='Input embedding file')
    parser.add_argument('--method', required=True, choices=['deepwalk', 'node2vec'], help='Method used for generating embedding')
    parser.add_argument('--labels', required=True, help='Labels file')
    parser.add_argument('--clustering', required=True, choices=['kmeans', 'sc'], help='The clustering algorithm to be used.')

    args = parser.parse_args()
    main(args)
