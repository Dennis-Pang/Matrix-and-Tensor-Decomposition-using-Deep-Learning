import torch
import torch.nn as nn
import torch.nn.functional as F

class VM2D(nn.Module):
    """
    
        Do 2D/1D convolutional decomposition of slices and fibers on the input 4D tensor ([B, H, W, D]).
        
    """
    def __init__(self, k, base_channels):
        super().__init__()
        self.k = k
        # Original style: 2-layer convolution + ReLU + pooling
        self.slice_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 2*base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4*base_channels, k, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Output: [batch, k, 1, 1]
        )
        self.fiber_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, 2*base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(2*base_channels, 4*base_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(4*base_channels, k, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # Output: [batch, k, 1]
        )

    def forward(self, x, k=None, slice_batch=32, fiber_batch=512):
        """
                
                x: [B, H, W, D]
                k: If specified, only the first k components are used for dimension-by-dimensional training
                Return: dict{'vector': [B, H, k_used], 'matrix': [B, W, D, k_used]}
                
        """
        B, H, W, D = x.shape
        r = k if k is not None else self.k

        # --- slices (each slice is [1, W, D]) ---
        x_slices = x.reshape(B*H, 1, W, D)
        
        # Pre-allocate slice output tensor, using complete k channels
        slice_output_shape = (B*H, self.k, 1, 1)
        slice_results = torch.zeros(slice_output_shape, device=x.device, dtype=x.dtype)
        
        # Write directly to preallocated memory
        for i in range(0, B*H, slice_batch):
            end_idx = min(i + slice_batch, B*H)
            slice_results[i:end_idx] = self.slice_conv(x_slices[i:end_idx])
        
        # Only take the first r dimensions
        vectors = slice_results.view(B, H, self.k)[:, :, :r]
        
        # Clean up intermediate results
        del x_slices, slice_results

        # --- fibers (each fiber is [1, H]), fiber is in the H direction
        x_fibers = x.permute(0,2,3,1).reshape(B*W*D, 1, H)  # [B*W*D, 1, H]
        
        # Pre-allocate the output tensor, using the complete k channels
        output_shape = (B*W*D, self.k, 1)
        matrices_full = torch.zeros(output_shape, device=x.device, dtype=x.dtype)

        # Write directly to preallocated memory
        for i in range(0, B*W*D, fiber_batch):
            end_idx = min(i + fiber_batch, B*W*D)
            matrices_full[i:end_idx] = self.fiber_conv(x_fibers[i:end_idx])
        
        # Only take the first r dimensions
        matrices = matrices_full.view(B, W, D, self.k)[:, :, :, :r]
        
        # Clean up intermediate results
        del x_fibers, matrices_full
        return {'vector': vectors, 'matrix': matrices}

    def reconstruct(self, vector, matrix, k=None):
        """
        vector: [B, H, k_used]
        matrix: [B, W, D, k_used]
        k: number of dimensions used (consistent with forward)
        return: [B, H, W, D]
        """
        return torch.einsum('bhk,bwdk->bhwd', vector, matrix)

class VM3D(nn.Module):
    """
        
        For inputs [B, H, W, D], VM2D decomposition is performed on the H/W/D directions (each direction can be covered).
        
    """
    def __init__(self, k, input_size, base_channels):
        super().__init__()
        H, W, D = input_size
        self.k = k
        self.base_channels = base_channels
        self.input_size = input_size
        self.net_H = VM2D(k, base_channels)
        self.net_W = VM2D(k, base_channels)
        self.net_D = VM2D(k, base_channels)

    def forward(self, X, k=None, slice_batch=32, fiber_batch=512):
        if X.ndim == 3:
            X = X.unsqueeze(0)
        B, H, W, D = X.shape
        r = k if k is not None else self.k

        # Dynamically adjust batch size to suit big data
        # For big data, increase the batch size to reduce the number of loops
        if H * W * D > 64**3:  # Adjustment when greater than 64
            fiber_batch = min(fiber_batch * 8, W * D // 16)  # Enlarge fiber_batch
            slice_batch = min(slice_batch * 4, H // 16)      # Enlarge slice_batch
            
        # H spindle: X as [B, H, W, D]
        out_H = self.net_H(X, r, slice_batch, fiber_batch)
        V_H, M_H = out_H['vector'], out_H['matrix']
        X_rec_H = self.net_H.reconstruct(V_H, M_H, r)
        
        # Cleaning the intermediate results of H direction
        del out_H

        # W spindle: X_W as [B, W, D, H]
        X_W = X.permute(0,2,3,1).contiguous()  # [B, W, D, H]
        out_W = self.net_W(X_W, r, slice_batch, fiber_batch)
        V_W, M_W = out_W['vector'], out_W['matrix']
        X_rec_W = self.net_W.reconstruct(V_W, M_W, r)
        X_rec_W = X_rec_W.permute(0,3,1,2).contiguous()  # Back to [B, H, W, D]
        
        # Clean the intermediate results of W direction
        del X_W, out_W

        # D spindle: X_D as [B, D, H, W]
        X_D = X.permute(0,3,1,2).contiguous()  # [B, D, H, W]
        out_D = self.net_D(X_D, r, slice_batch, fiber_batch)
        V_D, M_D = out_D['vector'], out_D['matrix']
        X_rec_D = self.net_D.reconstruct(V_D, M_D, r)
        X_rec_D = X_rec_D.permute(0,2,3,1).contiguous()  # Back to [B, H, W, D]
        
        # Clean the intermediate results of direction D
        del X_D, out_D

        # Add up to three directions
        X_rec = X_rec_H + X_rec_W + X_rec_D
        
        # Clean up intermediate results of reconstruction
        del X_rec_H, X_rec_W, X_rec_D

        return {
            'X_rec': X_rec,
            'V_H': V_H, 'M_H': M_H,
            'V_W': V_W, 'M_W': M_W,
            'V_D': V_D, 'M_D': M_D
        }

    def loss(self, X, output):
        if X.ndim == 3:
            X = X.unsqueeze(0)
        L_rec = F.mse_loss(output['X_rec'], X)
        return L_rec, torch.tensor(0.0, device=X.device), L_rec # no ortho loss

    def info(self):
        return f"VM3D(k={self.k}, base_channels={self.base_channels}, input_size={self.input_size})"