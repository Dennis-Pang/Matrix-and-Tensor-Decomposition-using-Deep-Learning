import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


def visualize_results(data, model1, model2, k, path):
    """
    
        Visual results:
        1. Save the reconstruction renderings (reconstruction.png)
        2. Save orthogonality graph (orthogonality.png)
        3. Save MSE/RelErr/OrthErr to mse.txt
    
        parameter:
        data: Input data (2D/3D Tensor)
        model1, model2: trained model
        k: rank
        path: save the path
        
    """
    os.makedirs(path, exist_ok=True)

    # ====== Raw data preparation =====
    if data.dim() == 3:  # (batch, H, W) -> Take the first one
        data = data[0]
    data_np = data.cpu().detach().numpy()

    # ==========================================================
    # 1. Reconstruction + MSE
    # ==========================================================
    plt.figure(figsize=(20, 5))

    # Original image
    plt.subplot(141)
    plt.imshow(data_np, cmap="gray")
    plt.axis("off")

    # --- SVD reconstruction ---
    U, S, Vh = torch.linalg.svd(data, full_matrices=False)
    svd_recon = (U[:, :k] @ torch.diag(S[:k])) @ Vh[:k, :]
    svd_recon_np = svd_recon.cpu().detach().numpy()
    svd_mse = np.mean((data_np - svd_recon_np) ** 2) / 65025
    plt.subplot(142)
    plt.imshow(svd_recon_np, cmap="gray")
    plt.axis("off")

    # --- Model1 Reconstruction ---
    with torch.no_grad():
        out1 = model1(data.unsqueeze(0).to(next(model1.parameters()).device), k)
        recon1 = out1["X_rec"][0].cpu().detach().numpy()
    model1_mse = np.mean((data_np - recon1) ** 2) / 65025
    plt.subplot(143)
    plt.imshow(recon1, cmap="gray")
    plt.axis("off")

    # --- Model2 Reconstruction ---
    with torch.no_grad():
        out2 = model2(data.unsqueeze(0).to(next(model2.parameters()).device), k)
        recon2 = out2["X_rec"][0].cpu().detach().numpy()
    model2_mse = np.mean((data_np - recon2) ** 2) / 65025
    plt.subplot(144)
    plt.imshow(recon2, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(path, "reconstruction.png"))
    plt.close()

    # ==========================================================
    # 2. Orthogonality
    # ==========================================================
    plt.figure(figsize=(25, 5))

    # ---- Model1 U/V ----
    U1, V1 = out1["U"][0], out1["V"][0]
    U1n = U1 / torch.norm(U1, dim=0, keepdim=True)
    V1n = V1 / torch.norm(V1, dim=0, keepdim=True)

    plt.subplot(141)
    plt.imshow((U1n.T @ U1n).cpu().numpy(), cmap="viridis_r", vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis("off")

    plt.subplot(142)
    plt.imshow((V1n.T @ V1n).cpu().numpy(), cmap="viridis_r", vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis("off")

    # ---- Model2 U/V ----
    U2, V2 = out2["U"][0], out2["V"][0]
    U2n = U2 / torch.norm(U2, dim=0, keepdim=True)
    V2n = V2 / torch.norm(V2, dim=0, keepdim=True)

    plt.subplot(143)
    plt.imshow((U2n.T @ U2n).cpu().numpy(), cmap="viridis_r", vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis("off")

    plt.subplot(144)
    plt.imshow((V2n.T @ V2n).cpu().numpy(), cmap="viridis_r", vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(path, "orthogonality.png"))
    plt.close()

    # ==========================================================
    # 3. Save MSE, RelErr and OrthErr to txt
    # ==========================================================
    def orth_err_normalized(X):
        # Column Normalization
        Xn = F.normalize(X, p=2, dim=0)
        r = Xn.shape[1]
        I = torch.eye(r, device=X.device)
        return torch.norm(Xn.T @ Xn - I, p="fro") / torch.norm(I, p="fro")


    # Orthogonality
    U1_err, V1_err = orth_err_normalized(U1).item(), orth_err_normalized(V1).item()
    U2_err, V2_err = orth_err_normalized(U2).item(), orth_err_normalized(V2).item()

    # RelErr (relative error ratio)
    model1_relerr = model1_mse / svd_mse
    model2_relerr = model2_mse / svd_mse

    with open(os.path.join(path, "mse.txt"), "w") as f:
        f.write(f"SVD MSE:          {svd_mse:.6f}\n")
        f.write(f"Model1 MSE:       {model1_mse:.6f}\n")
        f.write(f"Model1 RelErr:    {model1_relerr:.6f}\n")
        f.write(f"Model1 OrthErr U: {U1_err:.6f}, V: {V1_err:.6f}\n")
        f.write(f"Model2 MSE:       {model2_mse:.6f}\n")
        f.write(f"Model2 RelErr:    {model2_relerr:.6f}\n")
        f.write(f"Model2 OrthErr U: {U2_err:.6f}, V: {V2_err:.6f}\n")