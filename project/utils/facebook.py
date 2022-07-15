import os


def get_circles(root_path):
    files = os.listdir(root_path)
    circles = []
    
    for file in files:
        circle = file.split('.')[0]

        if circle not in circles:
            circles.append(circle)
    
    return circles


def circles_to_communities(circle_file):
    communities = {}
    with open(circle_file, 'r') as f:
        for circle, line in enumerate(f):
            if line.startswith('#'):
                continue
            else:
                nodes = list(map(int, line.strip().split('\t')[1:]))
                for node in nodes:
                    if circle not in communities:
                        communities[circle] = []
                    communities[circle].append(node)

    return communities
