import torch
import numpy as np

def batch_loader(Xs, ys, batch_size:int, seed:int): 
    if seed is not None:
        np.random.seed(seed)
        
    inputs, target = [x.clone() for x in Xs], [y.clone() for y in ys]
    input_size = [Xs[i].size(0) for i in range(len(Xs))]
    max_input_size = max(input_size)
    n_sources = len(Xs)
    
    for i in range(n_sources):
        r_order = np.arange(input_size[i])
        np.random.shuffle(r_order)
        inputs[i], target[i] = inputs[i][r_order, :, :], target[i][r_order]
        
    num_blocks = int(max_input_size/batch_size)
    for _ in range(num_blocks):
        xs, ys = [], []
        for i in range(n_sources):
            ridx = np.random.choice(input_size[i], batch_size)
            xs.append(inputs[i][ridx])
            ys.append(target[i][ridx])
            
        yield xs, ys  

def split_source_target(data: dict, test_idx: int, device):

    Xs, ys = [], []
    for sub in set(data['subject']):
        if sub == test_idx:
            Xt = torch.FloatTensor(data['X'][data['subject'] == test_idx]).to(device)
            yt = torch.FloatTensor(data['y'][data['subject'] == test_idx]).to(device)
          
        else:
            Xs.append(torch.FloatTensor(data['X'][data['subject'] == sub]).to(device))
            ys.append(torch.FloatTensor(data['y'][data['subject'] == sub]).to(device))
                
    return Xs, ys, Xt, yt