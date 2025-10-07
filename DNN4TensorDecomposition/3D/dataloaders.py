import torch
import numpy as np
import ast

def load_data(name):
    if name == 'skull':
        return load_skull_data()
    elif name == 'c60':
        return load_c60_data()
    elif name == 'hazelnut':
        return load_hazelnut_data()
    elif name == 'mnist':
        return load_mnist_data()
    
def load_mnist_data():
    path = r'E:\SVD\grey_data\3D_mnist.txt'
    with open(path, 'r') as f:
        txt = f.read()
    vol_list = ast.literal_eval(txt)
    vol_np = np.array(vol_list, dtype=np.float32)  # shape should be (16,16,16)
    data = torch.from_numpy(vol_np)            # shape: [16,16,16]
    return data, data.shape[1:], 'mnist'

def load_skull_data(
    path=r'E:\SVD\grey_data\Skull_res68x256x256_size1.0x1.0x1.0.npy',
    axis=None,              # None | 'H'|'W'|'D'
    as_rows=True,             # After unfolding, the axis is used as a row (True) or column (False)
):
    """
    
        Returns: tensor, input_size, data_name
        - If unfold is None: tensor shape [B,D,H,W] (maintain 4D volume data)
          input_size = (D,H,W)
        - If unfold in {'x','y','z'}:
            as_rows=True  -> [B, rows, cols]
            as_rows=False -> [B, cols, rows] (Put the axis into the column)
          If keep4d=True, then add channel in the first dimension -> [B,1,*,*]
          input_size = (rows, cols)
        
    """
    skull_data = np.load(path)
    tensor = torch.from_numpy(skull_data).float().unsqueeze(0) / 255.0  # [B=1,D,H,W]

    tensor = torch.rot90(tensor, k=1, dims=(2, 3))
    tensor = torch.rot90(tensor, k=1, dims=(1, 2)) # 256,68,256

    B, H, W, D = tensor.shape

    if axis is None:
        return tensor, (H, W, D), 'skull'

    if axis=="W":
        M = tensor.permute(0, 3, 1, 2).contiguous().view(B, W, D * H)  # [B,W,D*H]
        rows, cols = (W, D * H) if as_rows else (D * H, W)
        if not as_rows:
            M = M.transpose(1, 2)  # [B,D*H,W]
        name = 'skull_unfold_x'
    elif axis=="H":
        M = tensor.permute(0, 2, 1, 3).contiguous().view(B, H, D * W)  # [B,H,D*W]
        rows, cols = (H, D * W) if as_rows else (D * W, H)
        if not as_rows:
            M = M.transpose(1, 2)  # [B,D*W,H]
        name = 'skull_unfold_y'
    elif axis=="D":
        M = tensor.contiguous().view(B, D, H * W) 
        rows, cols = (D, H * W) if as_rows else (H * W, D)
        if not as_rows:
            M = M.transpose(1, 2)  # [B,H*W,D]
        name = 'skull_unfold_z'
    else:
        raise ValueError(f"unfold must be None/'x'/'y'/'z', got ")

    return M, (rows, cols), name

def load_c60_data():
    ball_path = r'E:\SVD\grey_data\C60Large.npy'
    data = np.load(ball_path)
    tensor = torch.from_numpy(data).float().unsqueeze(0)
    return tensor, tensor.shape[1:], 'c60'  

def load_hazelnut_data():
    p = r'E:\SVD\data\hazelnuts\hnut512_uint.raw'
    shape = (512, 512, 512)  
    dtype = np.uint8   
    with open(p, 'rb') as f:
        volume_np = np.fromfile(f, dtype=dtype)
    data = volume_np.reshape(shape)

    tensor = torch.from_numpy(data).float().unsqueeze(0) / 255.0
    return tensor, tensor.shape[1:], 'hazelnut'