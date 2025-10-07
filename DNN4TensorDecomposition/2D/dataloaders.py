import torch
import numpy as np

# def load_whale_data():
#     data = torch.from_numpy(np.load(r"E:\SVD\grey_data\whale_gray.npy")).float()
#     shape = data.shape[1:]
#     return data, shape, "whale"

def load_data(p):
    path = r"E:\SVD\grey_data\\" + str(p) + ".npy" 
    data = torch.from_numpy(np.load(path)).float().unsqueeze(0)
    shape = data.shape[1:]
    return data, shape, p