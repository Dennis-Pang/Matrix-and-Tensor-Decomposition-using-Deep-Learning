import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use DIP_1D from base_networks as Overparam1DGenerator

class DIP_1D(nn.Module):
    def __init__(self, L, output_channels, base_channels: int = 128,
                 in_channels: int = 1, depthwise_first: bool = False):
        """
        
                L: One-dimensional length (W)
                output_channels: Output channel = rank (R)
                in_channels: input number of noise channels; when WR input is to be made, set to R
                depthwise_first: If True, the first layer is grouped by in_channels (hard isolation of channels for each rank)
                
        """
        super().__init__()
        self.L = L
        self.base_c = base_channels
        self.output_channels = output_channels

        groups = in_channels if depthwise_first else 1

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, self.base_c, kernel_size=(3,1), stride=(2,1),
                      padding=(1,0), groups=groups),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1), mode='nearest'),
            nn.Conv2d(self.base_c, self.base_c, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.base_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.to_output = nn.Conv2d(self.base_c, self.output_channels, kernel_size=1, padding=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        
                z: [B, in_channels(=R), L, 1] -- WR input
                return: [B, L, R]
                
        """
        e1 = self.enc1(z)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bottleneck(e4) + e4

        d4 = self.dec4(b)
        if d4.size(2) != e3.size(2):
            d4 = F.interpolate(d4, size=e3.shape[-2:], mode='nearest')
        d4 = d4 + e3

        d3 = self.dec3(d4)
        if d3.size(2) != e2.size(2):
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode='nearest')
        d3 = d3 + e2

        d2 = self.dec2(d3)
        if d2.size(2) != e1.size(2):
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode='nearest')
        d2 = d2 + e1

        d1 = self.dec1(d2)
        if d1.size(2) != z.size(2):
            d1 = F.interpolate(d1, size=z.shape[-2:], mode='nearest')

        out = self.to_output(d1)                 # [B, R, L, 1] with output_channels=R
        return out.squeeze(3).transpose(1, 2)    # -> [B, L, R]

Overparam1DGenerator = DIP_1D

class CP_DIP(nn.Module):
    def __init__(self, input_size, k: int, base_channels: int = 128,
                 temperature: float = 0.0, weight: float = 0.0,
                 depthwise_first: bool = False):
        """
                
                input_size: (I, J, K)
                k: CP rank (R)
                depthwise_first: If True, it can hard-isolate each rank channel on the first layer (groups=R)
                
        """
        super().__init__()
        I, J, K = input_size
        self.I, self.J, self.K = I, J, K
        self.k = k
        self.base_c = base_channels
        self.temperature = temperature
        self.weight = weight

        # --- Key change: Number of noise channels = R=k (WR input) ---
        zU = torch.rand(1, k, I, 1) * 2 - 1   # (1,R,I,1)
        zV = torch.rand(1, k, J, 1) * 2 - 1   # (1,R,J,1)
        zW = torch.rand(1, k, K, 1) * 2 - 1   # (1,R,K,1)

        self.register_buffer('zU_base', zU)
        self.register_buffer('zV_base', zV)
        self.register_buffer('zW_base', zW)

        # Generator: Input Channel = R
        self.netU = Overparam1DGenerator(I, k, base_channels=self.base_c,
                                         in_channels=k, depthwise_first=depthwise_first)
        self.netV = Overparam1DGenerator(J, k, base_channels=self.base_c,
                                         in_channels=k, depthwise_first=depthwise_first)
        self.netW = Overparam1DGenerator(K, k, base_channels=self.base_c,
                                         in_channels=k, depthwise_first=depthwise_first)

    def forward(self, X_true: torch.Tensor) -> dict:
        """
        X_true: (B, I, J, K)
        returns: {'U':(B,I,R), 'V':(B,J,R), 'W':(B,K,R), 'X_rec':(B,I,J,K)}
        """
        device = next(self.parameters()).device
        B, Ii, Jj, Kk = X_true.shape
        assert (Ii, Jj, Kk) == (self.I, self.J, self.K)

        # Expand the noise to batch: (B,R,L,1)
        zU = self.zU_base.expand(B, -1, -1, -1).to(device)
        zV = self.zV_base.expand(B, -1, -1, -1).to(device)
        zW = self.zW_base.expand(B, -1, -1, -1).to(device)

        U = self.netU(zU)   # (B, I, R)
        V = self.netV(zV)   # (B, J, R)
        W = self.netW(zW)   # (B, K, R)

        # CP reconstruction: sum over R
        # bir, bjr, bcr -> bijk
        X_rec = torch.einsum('bir,bjr,bkr->bijk', U, V, W)

        return {'U': U, 'V': V, 'W': W, 'X_rec': X_rec}

    def orthogonality_loss(self, U, V, W):
        batch_size, _, r = U.shape
        loss = 0.0
        targets = torch.arange(r, device=U.device)
        for b in range(batch_size):
            U_n = F.normalize(U[b], p=2, dim=0)
            V_n = F.normalize(V[b], p=2, dim=0)
            W_n = F.normalize(W[b], p=2, dim=0)
            S_U = (U_n.t() @ U_n) * self.temperature
            S_V = (V_n.t() @ V_n) * self.temperature
            S_W = (W_n.t() @ W_n) * self.temperature
            loss += F.cross_entropy(S_U, targets)
            loss += F.cross_entropy(S_V, targets)
            loss += F.cross_entropy(S_W, targets)
        return loss

    def loss(self, X_true: torch.Tensor, output: dict):
        X_rec = output['X_rec']
        loss_recon = F.mse_loss(X_rec, X_true)
        U, V, W = output['U'], output['V'], output['W']
        loss_ortho = self.weight * self.orthogonality_loss(U, V, W)
        total = loss_recon + loss_ortho
        return loss_recon, loss_ortho, total

# class CP_DIP(nn.Module):
#     def __init__(self, input_size, k: int, base_channels: int = 128,
#                  temperature: float = 0, weight: float = 0.0):
#         """
#         I, J, K: Dimensions of the target 3D tensor X_true, all must be divisible by 16
#         r      : CP decomposition rank, must also be divisible by 16
#         base_channels: Number of hidden channels in Overparam1DGenerator, default 128
#         temperature: Temperature parameter used for orthogonality constraint
#         weight: Weight of the orthogonality constraint loss
#         """
#         super(CP_DIP, self).__init__()
#         I, J, K = input_size
#         # # -- Check if I,J,K,r are all divisible by 16 -- 
#         # assert I % 16 == 0, f"I (={I}) must be divisible by 16."
#         # assert J % 16 == 0, f"J (={J}) must be divisible by 16."
#         # assert K % 16 == 0, f"K (={K}) must be divisible by 16."

#         self.I = I
#         self.J = J
#         self.K = K
#         self.k = k
#         self.base_c = base_channels
#         self.temperature = temperature
#         self.weight = weight

#         # -- Sample noise for U, V, W once in __init__ and fix as buffers -- 
#         #    zU_base shape (1,1,I,  1)
#         zU = torch.rand(1, 1, I, 1) * 2 - 1
#         #    zV_base shape (1,1,J,  1)
#         zV = torch.rand(1, 1, J, 1) * 2 - 1
#         #    zW_base shape (1,1,K,  1)
#         zW = torch.rand(1, 1, K, 1) * 2 - 1

#         # Register as buffers, will not be updated during training
#         self.register_buffer('zU_base', zU)
#         self.register_buffer('zV_base', zV)
#         self.register_buffer('zW_base', zW)

#         # Define 3 Overparam1DGenerators: to generate U (I×r), V (J×r), W (K×r)
#         # Their input shapes are all (B,1,L), where L is I, J, K respectively; output (B,1,L)
#         self.netU = Overparam1DGenerator(I, k, base_channels=self.base_c)
#         self.netV = Overparam1DGenerator(J, k, base_channels=self.base_c)
#         self.netW = Overparam1DGenerator(K, k, base_channels=self.base_c)

#     def forward(self, X_true: torch.Tensor) -> dict:
#         """
#         X_true: True 3D tensor, shape (B, I, J, K)
#         Returns dict {
#           'U': (B, I, r),
#           'V': (B, J, r),
#           'W': (B, K, r),
#           'X_rec': (B, I, J, K)
#         }
#         """
#         device = next(self.parameters()).device
#         B, Ii, Jj, Kk = X_true.shape
#         assert Ii == self.I and Jj == self.J and Kk == self.K, (
#             f"Expected X_true shape (B,{self.I},{self.J},{self.K}), but got (B,{Ii},{Jj},{Kk})"
#         )

#         # -- 1) Expand zU_base → zU ∈ (B,1,I,r) -- 
#         zU = self.zU_base.expand(B, -1, -1, -1).to(device)  # (B,1,I,r)
#         U = self.netU(zU)        # -> (B,I,r)

#         # -- 2) Expand zV_base → zV ∈ (B,1,J,r) -- 
#         zV = self.zV_base.expand(B, -1, -1, -1).to(device)  # (B,1,J,r)
#         V = self.netV(zV)        # -> (B,J,r)

#         # -- 3) Expand zW_base → zW ∈ (B,1,K,r) -- 
#         zW = self.zW_base.expand(B, -1, -1, -1).to(device)  # (B,1,K,r)
#         W = self.netW(zW)        # -> (B,K,r)

#         X_rec = torch.einsum('bik,bjk,bck->bijc', U, V, W)

#         return {'U': U, 'V': V, 'W': W, 'X_rec': X_rec}

#     def orthogonality_loss(self, U, V, W):
#         """
#         Optional: Apply column orthogonality constraint to U, V, W for each batch
#         """
#         batch_size, _, r = U.shape  # r = rank
#         loss = 0.0
#         targets = torch.arange(r, device=U.device)
#         for b in range(batch_size):
#             # U orthogonality
#             U_n = F.normalize(U[b], p=2, dim=0)    # (I, r) normalize each column
#             S_U = torch.matmul(U_n.t(), U_n) * self.temperature  # (r, r)
#             loss += F.cross_entropy(S_U, targets)

#             # V orthogonality
#             V_n = F.normalize(V[b], p=2, dim=0)
#             S_V = torch.matmul(V_n.t(), V_n) * self.temperature  # (r, r)
#             loss += F.cross_entropy(S_V, targets)

#             # W orthogonality
#             W_n = F.normalize(W[b], p=2, dim=0)
#             S_W = torch.matmul(W_n.t(), W_n) * self.temperature  # (r, r)
#             loss += F.cross_entropy(S_W, targets)

#         return loss

#     def loss(self, X_true: torch.Tensor, output: dict):
#         """
#         Calculate reconstruction + optional orthogonality constraint loss
#         """
#         X_rec = output['X_rec']
#         loss_recon = F.mse_loss(X_rec, X_true)  # Reconstruction loss
#         U, V, W = output['U'], output['V'], output['W']
#         loss_ortho = self.weight * self.orthogonality_loss(U, V, W)
#         total = loss_recon + loss_ortho
#         return loss_recon, loss_ortho, total

#     def info(self) -> str:
#         s = (
#             f"DeepTensor3D:\n"
#             # f"  I={self.I}, J={self.J}, K={self.K},  r={self.r}, base_channels={self.base_c}\n"
#             # f"  zU_base.shape = {tuple(self.zU_base.shape)}\n"
#             # f"  zV_base.shape = {tuple(self.zV_base.shape)}\n"
#             # f"  zW_base.shape = {tuple(self.zW_base.shape)}\n"
#             # f"---- netU structure ----\n{self.netU}\n"
#             # f"---- netV structure ----\n{self.netV}\n"
#             # f"---- netW structure ----\n{self.netW}\n"
#         )
#         return s