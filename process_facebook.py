import os
import json
import torch as th
from project.utils.facebook import get_circles, circles_to_communities
from project.utils.utils import edges_file_to_adj, features_file_to_tensor


def main():
    circles = get_circles('data/raw/facebook')
    
    for circle in circles:
        circle_file = f'data/raw/facebook/{circle}.circles'
        communities = circles_to_communities(circle_file)
        adj = edges_file_to_adj(f'data/raw/facebook/{circle}.edges')
        nfeats = features_file_to_tensor(f'data/raw/facebook/{circle}.feat', adj.shape[0])
        
        os.makedirs(f'data/facebook/{circle}', exist_ok=True)

        with open(f'data/facebook/{circle}/communities.json', 'w') as f:
            json.dump(communities, f)

        th.save(adj, f'data/facebook/{circle}/adj.th')
        th.save(nfeats, f'data/facebook/{circle}/nfeats.th')        


if __name__ == '__main__':
    main()
