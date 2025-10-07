import torch
import torch.nn as nn

class Fiber(nn.Module):
    def __init__(self, k, base_channels):
        super().__init__()
        self.k = k
        self.slice_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2*base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)   # Output [batch, 4*base_channels, 1, 1]
        )
        # The last fully connected layer is explicitly written
        self.fc = nn.Linear(4*base_channels, k)

    def forward(self, slices, slice_batch=32):
        """
        
                Args:
                    slices: [total_slices, 1, H, W] - All slices in a single direction
                    slice_batch: batch size
                Returns:
                    [total_slices, k] - K-dimensional characteristics of each slice
                
        """
        total_slices = slices.size(0)
        results = torch.zeros(total_slices, self.k, device=slices.device, dtype=slices.dtype)

        for i in range(0, total_slices, slice_batch):
            end_idx = min(i + slice_batch, total_slices)
            batch_output = self.slice_conv(slices[i:end_idx])  # [batch, 4*base_channels, 1, 1]
            batch_output = batch_output.view(batch_output.size(0), -1)  # [batch, 4*base_channels]
            batch_output = self.fc(batch_output)  # [batch, k]
            results[i:end_idx] = batch_output

        return results


class CP(nn.Module):
    def __init__(self, k, base_channels):
        super().__init__()
        self.k = k
        # Three independent networks
        self.mode1_net = Fiber(k, base_channels)  # Processing [J,K] slices
        self.mode2_net = Fiber(k, base_channels)  # Processing [I,K] slices
        self.mode3_net = Fiber(k, base_channels)  # Handling [I,J] slices

    def forward(self, x, slice_batch=32):
        """
        
                Args:
                    x: [B, I, J, K] 3D tensor
                    slice_batch: batch size to avoid memory explosion
                Returns:
                    dict with X_rec, U1, U2, U3
                
        """
        B, I, J, K = x.shape

        # Mode-1: I slices of [J,K]
        mode1_input = x.view(B*I, 1, J, K)
        mode1_output = self.mode1_net(mode1_input, slice_batch)  # [B*I, k]
        A1 = mode1_output.view(B, I, self.k)
        del mode1_input, mode1_output

        # Mode-2: J slices of [I,K]
        mode2_input = x.permute(0, 2, 1, 3).contiguous().view(B*J, 1, I, K)
        mode2_output = self.mode2_net(mode2_input, slice_batch)  # [B*J, k]
        A2 = mode2_output.view(B, J, self.k)
        del mode2_input, mode2_output

        # Mode-3: K slices of [I,J]
        mode3_input = x.permute(0, 3, 1, 2).contiguous().view(B*K, 1, I, J)
        mode3_output = self.mode3_net(mode3_input, slice_batch)  # [B*K, k]
        A3 = mode3_output.view(B, K, self.k)
        del mode3_input, mode3_output

        # Reconstruction
        X_rec = torch.einsum('bir,bjr,bkr->bijk', A1, A2, A3)
        return {"X_rec": X_rec, "U1": A1, "U2": A2, "U3": A3}

    def loss(self, X, output):
        X_rec = output["X_rec"]
        mse = nn.functional.mse_loss(X_rec, X)
        return mse, mse, mse