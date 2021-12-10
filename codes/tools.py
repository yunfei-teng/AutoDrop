import os
import math
import torch

def pca_analysis(input, k = 5, q = 6, centered = True, normlaized=True):
    # 1. obtain projections on top k and the other principle components
    q = min([q, input.size(-2), input.size(-1)])
    assert q >= k, 'q is an overestimation of k so q should be no-less than k'
    _, _, e_vecs = torch.pca_lowrank(input, q, center=centered, niter=10)
    e_vecs = e_vecs[:, :k]

    # 2. absolute (and normalize) explained variance
    top_pcs = input.mm(e_vecs)
    other_pcs = (input - input.mm(e_vecs).mm(e_vecs.t())).norm(dim=1, keepdim=True)
    e_vars = torch.cat((top_pcs, other_pcs), dim = 1).abs()
    if normlaized:
        e_vars = e_vars / e_vars.norm(dim=1, keepdim=True)
        e_vars = e_vars** 2 # Are we normalizing to 1?
    return e_vars, e_vecs

def calculate_angle(v1, v2):
    epsilon = 1e-16
    dot_production = (v1*v2).sum()/(v1.norm()*v2.norm()+1e-8)
    range = torch.clamp(dot_production, -1.0+epsilon, 1.0-epsilon)
    return (torch.acos(range)/math.pi* 180).item()

def mkdir(path):
    '''create a single empty directory if it didn't exist
    Parameters: path (str) -- a single directory path'''
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    '''create empty directories if they don't exist
    Parameters: paths (str list) -- a list of directory paths'''
    # rmdirs(paths)
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def rmdirs(paths):
    if os.path.exists(paths):
        for file in os.listdir(paths): 
            file_path = os.path.join(paths, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(paths)