import torch
import tensorly as tl
from tensorly.decomposition import parafac, tucker
import ipyvolume as ipv
import ipywidgets as widgets
import numpy as np
import os
import matplotlib.pyplot as plt


tl.set_backend('pytorch')

def save_ipyvolume_html(filename, figure=None):
    if figure is None:
        figure = ipv.gcf()
    ipv.save(filename)
    print(f"[OK] Saved interactive HTML to: {filename}")

def save_mse_to_txt(mse_dict, save_path):
    """
    Save MSE results to a txt file.
    
    Parameters:
    mse_dict: A dictionary containing MSE values for each method.
    save_path: The path to save the file.
    """
    filepath = os.path.join(save_path, 'reconstruction_mse.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Reconstruction MSE Loss Results:\n")
        f.write("=" * 30 + "\n")
        for method, mse in mse_dict.items():
            f.write(f"{method}: {mse:.6f}\n")
    print(f"[OK] Saved MSE results to: {filepath}")

def tucker_decomposition(data, k):
    B = data.shape[0]
    X_rec = torch.zeros_like(data)
    
    # Set the rank for each dimension
    ranks = [k, k, k]
    
    for b in range(B):
        # Tucker decomposition
        core, factors = tucker(data[b], rank=ranks, init='random', tol=1e-4)
        
        # Reconstruction
        X_rec[b] = tl.tucker_to_tensor((core, factors))
    
    return {
        'X_rec': X_rec,
        'core': core,
        'factors': factors
    }

def cp_decomposition(data, k):
    """
    Perform CP decomposition using TensorLy with added robustness.
    
    Parameters:
    data: Input data, a tensor of shape [B, D, H, W].
    k: The rank of the decomposition.
    
    Returns:
    dict: A dictionary containing the reconstruction results.
    """
    B = data.shape[0]
    X_rec = torch.zeros_like(data)
    
    for b in range(B):
        try:
            # Data preprocessing: standardization
            tensor = data[b]
            mean_val = tensor.mean()
            std_val = tensor.std()
            if std_val > 0:
                tensor = (tensor - mean_val) / std_val
            
            # CP decomposition, using random initialization instead of svd
            weights, factors = parafac(
                tensor,
                rank=k,
                n_iter_max=500,
                init='random',  # Use random initialization
                tol=1e-4,      # Relax convergence conditions
                random_state=42,
                verbose=0
            )
            
            # Reconstruction
            X_rec_temp = tl.cp_to_tensor((weights, factors))
            
            # Restore original range
            if std_val > 0:
                X_rec_temp = X_rec_temp * std_val + mean_val
            
            X_rec[b] = X_rec_temp
            
        except Exception as e:
            print(f"Warning: Decomposition failed for batch {b}, using a zero tensor instead. Error: {str(e)}")
            X_rec[b] = torch.zeros_like(data[b])
            weights = None
            factors = None
            
    return {
        'X_rec': X_rec,
        'weights': weights,
        'factors': factors
    }

def visualize_3d_results(original_data, model_rec, k, threshold=0.0, save_path=None):
    # Traditional decompositions
    tucker_rec = tucker_decomposition(original_data, k)['X_rec']
    cp_rec = cp_decomposition(original_data, k)['X_rec']

    original = original_data[0].detach().cpu().numpy()
    tucker = tucker_rec[0].detach().cpu().numpy()
    cp = cp_rec[0].detach().cpu().numpy()
    model = model_rec[0].detach().cpu().numpy()

    datas = [original, tucker, cp, model]
    titles = ['Original', 'Tucker', 'CP', 'Model']
    mse_errors = [np.mean((original - d) ** 2) for d in datas]
    mse_dict = {
        'Tucker': mse_errors[1],
        'CP': mse_errors[2],
        'Model': mse_errors[3]
    }
    if save_path:
        save_mse_to_txt(mse_dict, save_path)
    widgets_list = []

    for data, title, mse in zip(datas, titles, mse_errors):
        ipv.figure()
        
        # Threshold filtering: set voxels below the threshold to 0 to avoid affecting the display
        data_filtered = np.copy(data)
        data_filtered[data_filtered < threshold] = 0.0
        
        # Display voxel data directly with volshow
        ipv.volshow(data_filtered, level=threshold)
        
        # Set axis ranges (optional, volshow defaults to full volume display)
        ipv.xlim(0, data.shape[2])
        ipv.ylim(0, data.shape[1])
        ipv.zlim(0, data.shape[0])

        canvas = ipv.gcf()
        text = widgets.HTML(f"<b>{title}</b><br>MSE: {mse:.6f}")
        widgets_list.append(widgets.VBox([text, canvas]))

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, f"3d_{title.lower()}.html")
            save_ipyvolume_html(filepath, ipv.gcf())
            print(f"Saved: {filepath}")

    return widgets.HBox(widgets_list)


def ortho_vis(output, path):
    # Display the orthogonality of the model's U matrix
    plt.figure(figsize=(15, 5))
    
    # Orthogonality of the model's U matrix
    with torch.no_grad():
        # Handle shape differences in different model outputs
        U = output['U1']
        V = output['U2']
        W = output['U3']
        
        # If U and V are 3D tensors, take the first sample
        if U.dim() == 3:
            U = U[0]
        if V.dim() == 3:
            V = V[0]
        if W.dim() == 3:
            W = W[0]        
        
        U_norm = U / torch.norm(U, dim=0, keepdim=True)
        U_ortho = torch.matmul(U_norm.t(), U_norm)
        plt.subplot(131)  
        plt.imshow(U_ortho.cpu().detach().numpy(), cmap='viridis_r', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Model U Orthogonality')
        
        # Orthogonality of the model's V matrix
        # Calculate the cosine similarity matrix
        V_norm = V / torch.norm(V, dim=0, keepdim=True)
        V_ortho = torch.matmul(V_norm.t(), V_norm)
        plt.subplot(132)  
        plt.imshow(V_ortho.cpu().detach().numpy(), cmap='viridis_r', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Model V Orthogonality')
        
        plt.tight_layout()

        W_norm = W / torch.norm(W, dim=0, keepdim=True)
        W_ortho = torch.matmul(W_norm.t(), W_norm)
        plt.subplot(133)  
        plt.imshow(W_ortho.cpu().detach().numpy(), cmap='viridis_r', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Model W Orthogonality')
        
        plt.tight_layout()

    plt.savefig(os.path.join(path, 'orthogonality_comparison.png'))
    plt.close()
