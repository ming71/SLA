import torch
import numpy as np

def hyp_parse(hyp_path):
    hyp = {}
    keys = [] 
    with open(hyp_path,'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0 : continue
            v = line.strip().split(':')
            try: 
                hyp[v[0]] = float(''.join(v[1].split()))
            except:
                hyp[v[0]] = eval(''.join(v[1].split()))
            keys.append(v[0])
        f.close()
    return hyp



def soft_weight(x):
    tp = 0.5
    sharpness = 20
    peak = 0.2
    return 1- peak / (1 + np.exp(-sharpness * (x - tp)))