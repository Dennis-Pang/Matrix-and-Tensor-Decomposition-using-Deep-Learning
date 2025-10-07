import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Three-layer MLP (default Softplus, hidden=128)=====
def mlp(in_dim, hidden=128, out_dim=1, depth=3, act=nn.Softplus):
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class NeuralSVD_MLP(nn.Module):
    """
    
        Sequential Nesting with MSE
        
    """
    def __init__(self,
                 input_size,  # (H, W)
                 k,
                 hidden=128,
                 depth=3,
                 use_fourier: bool = True,
                 ff_num_scales: int = 4,
                 ff_num_freq_per_scale: int = 256,
                 ff_base_sigma: float = 10.0,
                 ff_gamma: float = 0.5,
                 ff_include_input: bool = True,
                 use_sequential_nesting: bool = True):
        super().__init__()
        self.H, self.W = input_size
        self.k = k
        self.use_fourier = use_fourier
        self.use_sequential_nesting = use_sequential_nesting

        # norm coord([-1,1])
        i = torch.linspace(-1.0, 1.0, self.H).view(self.H, 1)  # [H,1]
        j = torch.linspace(-1.0, 1.0, self.W).view(self.W, 1)  # [W,1]
        self.register_buffer("row_coords", i)
        self.register_buffer("col_coords", j)

        # Fourier feature
        if use_fourier:
            from math import sqrt
            self.ff_row = self._make_fourier(ff_num_scales, ff_num_freq_per_scale,
                                             ff_base_sigma, ff_gamma, ff_include_input)
            self.ff_col = self._make_fourier(ff_num_scales, ff_num_freq_per_scale,
                                             ff_base_sigma, ff_gamma, ff_include_input)
            in_dim = self.ff_row.out_dim
        else:
            self.ff_row = None
            self.ff_col = None
            in_dim = 1

        self.row_heads = nn.ModuleList([mlp(in_dim, hidden, 1, depth) for _ in range(k)])
        self.col_heads = nn.ModuleList([mlp(in_dim, hidden, 1, depth) for _ in range(k)])

    def _make_fourier(self, num_scales, num_freq_per_scale, base_sigma, gamma, include_input):
        class FourierFeatures1D(nn.Module):
            def __init__(self, num_scales, num_freq_per_scale, base_sigma, gamma, include_input):
                super().__init__()
                self.include_input = include_input
                sigmas = [base_sigma * (gamma ** s) for s in range(num_scales)]
                Bs = []
                for sigma in sigmas:
                    B_s = torch.randn(1, num_freq_per_scale) * sigma
                    Bs.append(B_s)
                B = torch.cat(Bs, dim=1)
                self.register_buffer("B", B)
                self.out_dim = (1 if include_input else 0) + 2 * B.shape[1]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                proj = x @ self.B
                if self.include_input:
                    return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
                else:
                    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return FourierFeatures1D(num_scales, num_freq_per_scale, base_sigma, gamma, include_input)

    def forward(self, x, k=None):
        B = x.shape[0]
        r_req = self.k if k is None else int(k)
        r = max(0, min(self.k, r_req))

        r_in = self.row_coords
        c_in = self.col_coords
        if self.use_fourier:
            r_in = self.ff_row(r_in)  # [H, Din]
            c_in = self.ff_col(c_in)  # [W, Din]

        U_cols, V_cols = [], []
        for ell in range(r):
            u_col = self.row_heads[ell](r_in)   # [H,1]
            v_col = self.col_heads[ell](c_in)   # [W,1]
            U_cols.append(u_col)
            V_cols.append(v_col)

        if r == 0:
            U = x.new_zeros(B, self.H, 0)
            V = x.new_zeros(B, self.W, 0)
            X_rec = x.new_zeros(B, self.H, self.W)
            return {"U": U, "V": V, "X_rec": X_rec, "uv_list": []}

        U = torch.cat(U_cols, dim=1).unsqueeze(0).expand(B, -1, -1)  # [B,H,r]
        V = torch.cat(V_cols, dim=1).unsqueeze(0).expand(B, -1, -1)  # [B,W,r]
        X_rec = torch.bmm(U, V.transpose(1, 2))                      # [B,H,W]

        uv_list = [(U[:, :, i], V[:, :, i]) for i in range(r)]
        return {"U": U, "V": V, "X_rec": X_rec, "uv_list": uv_list}

    def _compute_sequential_nesting_loss(self, x, uv_list):
        """
        
                Sequential Nesting MSE Accumulation
                
        """
        B, H, W = x.shape
        k = len(uv_list)

        if k == 0:
            mse = F.mse_loss(torch.zeros_like(x), x)
            return mse, torch.zeros_like(mse), mse

        # --------- 1) MSE for non-nested monitoring --------
        U_full = torch.stack([u for (u, _) in uv_list], dim=-1)  # [B,H,k]
        V_full = torch.stack([v for (_, v) in uv_list], dim=-1)  # [B,W,k]
        X_rec_full = U_full @ V_full.transpose(-2, -1)           # [B,H,W]
        monitor_mse = F.mse_loss(X_rec_full, x)

        # --------- 2) Nested MSE ---------
        total_mse_nested = x.new_tensor(0.0)
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
            U_ell = torch.stack(U_cols, dim=-1)
            V_ell = torch.stack(V_cols, dim=-1)
            X_rec_ell = U_ell @ V_ell.transpose(-2, -1)
            mse_ell = F.mse_loss(X_rec_ell, x)
            total_mse_nested += mse_ell

        return monitor_mse, torch.zeros_like(monitor_mse), total_mse_nested

    def loss(self, x, output):
        if self.use_sequential_nesting:
            return self._compute_sequential_nesting_loss(x, output["uv_list"])
        else:
            mse = F.mse_loss(output["X_rec"], x)
            return mse, torch.zeros_like(mse), mse