import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFC(nn.Module):

   def __init__(self, k, base_channels, input_size, temperature=0, weight=0.0, 
                use_sequential_nesting=True):
       super(CNNFC, self).__init__()
       self.k = k
       self.base_channels = base_channels
       self.H, self.W = input_size # assume square
       self.temperature = temperature
       self.weight = weight
       self.use_sequential_nesting = use_sequential_nesting
       
       self.conv = nn.Sequential(
           nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(base_channels, 2*base_channels, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(4*base_channels, 8*base_channels, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Flatten()
       )
           
       fc_input_size = 8 * base_channels * (self.H // 16) * (self.W // 16)   
               
       self.fc_list = nn.ModuleList([
           nn.Linear(fc_input_size, self.H + self.W) for _ in range(k)
       ])

   def forward(self, x, k=None):
        # x: [B, H, W]
        B = x.size(0)
        r = k if k is not None else self.k

        x_in = x.unsqueeze(1)                # [B,1,H,W]
        feats = self.conv(x_in)              # [B, C_flat]

        uv_list = []
        for i in range(r):
            vec = self.fc_list[i](feats)     # [B, H+W]
            u_i = vec[:, :self.H]            # [B, H]
            v_i = vec[:, self.H:self.H+self.W]  # [B, W]
            uv_list.append((u_i, v_i))

        if r == 0:
            U = x.new_zeros(B, self.H, 0)
            V = x.new_zeros(B, self.W, 0)
            X_rec = x.new_zeros(B, self.H, self.W)
            return {'X_rec': X_rec, 'U': U, 'V': V, 'uv_list': []}

        U = torch.stack([pair[0] for pair in uv_list], dim=-1)  # [B, H, r]
        V = torch.stack([pair[1] for pair in uv_list], dim=-1)  # [B, W, r]

        X_rec = torch.matmul(U, V.transpose(-2, -1))            # [B, H, W]
        return {'X_rec': X_rec, 'U': U, 'V': V, 'uv_list': uv_list}


   def orthogonality_loss(self, U, V):
       batch_size, _, r = U.shape   # activated r
       loss = 0.0
       targets = torch.arange(r, device=U.device)
       for b in range(batch_size):
           U_n = F.normalize(U[b], p=2, dim=0)
           S_U = torch.matmul(U_n.t(), U_n) * self.temperature
           loss += F.cross_entropy(S_U, targets)
           V_n = F.normalize(V[b], p=2, dim=0)
           S_V = torch.matmul(V_n.t(), V_n) * self.temperature
           loss += F.cross_entropy(S_V, targets)
       return loss

   def _compute_sequential_nesting_loss(self, x, uv_list):
        B, H, W = x.shape
        k = len(uv_list)

        if k == 0:
            mse = F.mse_loss(torch.zeros_like(x), x)
            return mse, torch.zeros_like(mse), mse

        # --------- 1) MSE for non-necked monitoring (final rank-k reconstruction) --------
        U_full = torch.stack([u for (u, _) in uv_list], dim=-1)  # [B,H,k]
        V_full = torch.stack([v for (_, v) in uv_list], dim=-1)  # [B,W,k]
        X_rec_full = U_full @ V_full.transpose(-2, -1)           # [B,H,W]
        monitor_mse = F.mse_loss(X_rec_full, x)                  # Scalar

        # --------- 2) Nested MSE layer by layer + orthogonal loss ---------
        total_mse_nested = x.new_tensor(0.0)
        total_ortho = x.new_tensor(0.0)

        for ell in range(1, k + 1):
            U_cols, V_cols = [], []
            for j in range(ell):
                u_j, v_j = uv_list[j]
                if j < ell - 1:
                    U_cols.append(u_j.detach())
                    V_cols.append(v_j.detach())
                else:
                    U_cols.append(u_j)
                    V_cols.append(v_j)

            U_ell = torch.stack(U_cols, dim=-1)               # [B,H,ell]
            V_ell = torch.stack(V_cols, dim=-1)               # [B,W,ell]
            X_rec_ell = U_ell @ V_ell.transpose(-2, -1)       # [B,H,W]

            mse_ell = F.mse_loss(X_rec_ell, x)
            ortho_ell = self.orthogonality_loss(U_ell, V_ell)

            total_mse_nested += mse_ell
            total_ortho += ortho_ell

        total_loss = total_mse_nested + self.weight * total_ortho
        return monitor_mse, total_ortho, total_loss


   def loss(self, x, output):
       if self.use_sequential_nesting:
           # Loss with Sequential Nesting
           return self._compute_sequential_nesting_loss(x, output['uv_list'])
       else:
           # Use standard loss
           recon = output['X_rec']
           loss_recon = F.mse_loss(x, recon)
           loss_ortho = self.weight * self.orthogonality_loss(output['U'], output['V'])
           total_loss = loss_recon + loss_ortho
           return loss_recon, loss_ortho, total_loss