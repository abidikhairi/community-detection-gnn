import subprocess
from typing import Dict
from collections import OrderedDict


def exec_onmi(ground_truth_path, y_hat_path, onmi_exec = '/usr/bin/onmi'):

    res = subprocess.run([onmi_exec, ground_truth_path, y_hat_path], stdout=subprocess.PIPE)
    lines = res.stdout.decode('utf-8').splitlines()
    lines = list(map(lambda line: line.split('\t'), lines))

    result = {key: round(float(value) * 100, 4) for key, value in dict(lines).items()}
    
    return result


def communities_to_onmi(communities: Dict, output):
    sorted_communities = OrderedDict(sorted(communities.items(), key=lambda item: item[0]))
    with open(output, 'w') as stream:
        for idx, (_, nodes) in enumerate(sorted_communities.items()):
            line = ' '.join(map(lambda n: str(n), nodes))
            stream.write(line)
            
            if idx != len(communities) -1:
                stream.write('\n')
