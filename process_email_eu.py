import os
import torch as th
import numpy as np
from project.utils.utils import edges_file_to_adj


def main():
    labels = np.loadtxt('data/raw/email/department-labels.txt', dtype=np.int64, delimiter=' ')

    labels = th.from_numpy(labels[:, 1])
    adj = edges_file_to_adj('data/raw/email/email-Eu.txt')
    
    os.makedirs('data/email', exist_ok=True)

    th.save(labels, 'data/email/labels.th')
    th.save(adj, 'data/email/adj.th')


if __name__ == '__main__':
    main()
