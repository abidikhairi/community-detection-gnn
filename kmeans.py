import argparse
import torch as th
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score


def main(args):
    method = args.method
    
    features = th.load(args.input).numpy()
    labels = th.load(args.labels).numpy()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.15)

    num_clusters = max(labels) + 1

    model = KMeans(n_clusters=num_clusters)

    model.fit(x_train)


    y_hat = model.predict(x_test)

    nmi = normalized_mutual_info_score(y_test, y_hat)
    ami = adjusted_mutual_info_score(y_test, y_hat)

    print(f'Embedding Method: {method}')
    print(f'Clustering Algorithm K-Means')
    print(f'NMI score: {nmi*100:.2f} %')
    print(f'AMI score: {ami*100:.2f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='Input embedding file')
    parser.add_argument('--method', required=True, choices=['deepwalk', 'node2vec'], help='Method used for generating embedding')
    parser.add_argument('--labels', required=True, help='Labels file')

    args = parser.parse_args()
    main(args)
