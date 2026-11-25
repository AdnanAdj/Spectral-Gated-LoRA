"""
Core PEFT modules for Vision Transformers (timm ViT / DINOv2):
  - LoRABranch        (Hu et al., ICLR 2022)
  - DoRAQKV           (Liu et al., ICML 2024)
  - FourierFTBranch   (Gao et al., ICML 2024)
  - SGLoRABranch      (Proposed: Phase-Preserving Radial Spectral-Gated LoRA)
  - inject_peft()     (Unified injector for timm VisionTransformer blocks)
"""

import math
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_radial_masks(
    grid_h: int, grid_w: int, num_bands: int = 4, sigma_scale: float = 0.65
) -> torch.Tensor:
    """
    Precompute K smooth, non-negative Gaussian radial frequency masks M_k(u, v)
    over the 2D rFFT plane of shape (K, grid_h, grid_w // 2 + 1) such that
    sum_k M_k(u, v) == 1 at every frequency coordinate (u, v).
    """
    fy = torch.fft.fftfreq(grid_h).unsqueeze(1)        # (H, 1) in [-0.5, 0.5)
    fx = torch.fft.rfftfreq(grid_w).unsqueeze(0)       # (1, W//2 + 1) in [0, 0.5]
    rho = torch.sqrt(fy ** 2 + fx ** 2) / math.sqrt(0.5 ** 2 + 0.5 ** 2)  # [0, 1]

    if num_bands == 1:
        return torch.ones(1, grid_h, grid_w // 2 + 1, dtype=torch.float32)

    centers = torch.linspace(0.0, 1.0, num_bands)
    sigma = (1.0 / (num_bands - 1)) * sigma_scale
    masks = torch.stack(
        [torch.exp(-0.5 * ((rho - c) / sigma) ** 2) for c in centers], dim=0
    )
    masks = masks / (masks.sum(dim=0, keepdim=True) + 1e-8)
    return masks  # (K, H, W//2 + 1)


class LoRABranch(nn.Module):
    """Standard Low-Rank Adaptation branch: (alpha/r) * B(A(dropout(x)))."""

    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class FourierFTBranch(nn.Module):
    """
    Simplified FourierFT (Gao et al., ICML 2024): parameterizes the weight update
    matrix Delta_W in R^(dim x dim) using n sparse 2D DFT spectral entries.
    """

    def __init__(self, dim: int, n_freqs: int = 1000, scaling: float = 300.0, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.n_freqs = min(n_freqs, dim * dim)
        self.scaling = scaling
        g = torch.Generator()
        g.manual_seed(seed)
        idx = torch.randperm(dim * dim, generator=g)[: self.n_freqs]
        self.register_buffer("rows", idx // dim)
        self.register_buffer("cols", idx % dim)
        self.spectrum = nn.Parameter(torch.zeros(self.n_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build sparse spectral plane in float32 for numerical stability in AMP
        spec = torch.zeros(self.dim, self.dim, device=x.device, dtype=torch.float32)
        spec[self.rows, self.cols] = self.spectrum.float()
        delta_w = torch.fft.ifft2(spec).real * self.scaling
        return F.linear(x, delta_w.to(x.dtype))


class SGLoRABranch(nn.Module):
    """
    Proposed Spectral-Gated Low-Rank Adaptation (SG-LoRA):
      1. Down-project tokens X -> Z = A(X) in R^(B, N, r)
      2. Split prefix tokens (e.g. [CLS] + optional registers) and 2D spatial patch grid
      3. Apply 2D rFFT along spatial patch grid (H_p, W_p)
      4. Compute strictly positive real-valued radial band gate:
           G(u,v,c) = 1 + tanh( sum_k M_k(u,v) * (w_{k,c} + delta_w_{b,k,c}(z_cls)) )
         which strictly preserves 2D Fourier phase while modulating spectral amplitude.
      5. Apply 2D irFFT, re-concatenate prefix tokens, and up-project via zero-init B.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (16, 16),
        num_prefix_tokens: int = 1,
        num_bands: int = 4,
        dynamic: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.scaling = alpha / rank
        self.grid_h, self.grid_w = grid_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_bands = num_bands
        self.dynamic = dynamic

        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, dim, bias=False)

        # Precomputed non-trainable radial spectral masks: (K, H_p, W_p // 2 + 1)
        self.register_buffer("radial_masks", build_radial_masks(self.grid_h, self.grid_w, num_bands))

        # Static learnable radial band prior w in R^(K, r) initialized to 0 => G == 1
        self.band_weights = nn.Parameter(torch.zeros(num_bands, rank))

        # Optional [CLS]-conditioned dynamic residual band gate
        if self.dynamic:
            self.cls_norm = nn.LayerNorm(rank)
            self.cls_gate = nn.Linear(rank, num_bands * rank, bias=True)
            nn.init.zeros_(self.cls_gate.weight)
            nn.init.zeros_(self.cls_gate.bias)

        # Standard LoRA initialization: Kaiming on A, exact zeros on B
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def compute_band_logits(self, z_cls: torch.Tensor) -> torch.Tensor:
        """Return effective band logits of shape (B, K, r) for logging or gating."""
        B = z_cls.shape[0]
        w = self.band_weights.unsqueeze(0).expand(B, -1, -1)  # (B, K, r)
        if self.dynamic:
            delta_w = self.cls_gate(self.cls_norm(z_cls)).view(B, self.num_bands, self.rank)
            return w + delta_w
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d), where N = num_prefix_tokens + H_p * W_p
        B, N, _ = x.shape
        orig_dtype = x.dtype

        z = self.lora_A(self.dropout(x))  # (B, N, r)
        z_prefix = z[:, : self.num_prefix_tokens, :]  # (B, P, r)
        z_patch = z[:, self.num_prefix_tokens :, :]   # (B, H_p*W_p, r)

        # Ensure exact spatial grid match (supports dynamic resolution fallback if needed)
        H_p, W_p = self.grid_h, self.grid_w
        assert z_patch.shape[1] == H_p * W_p, (
            f"Patch token count {z_patch.shape[1]} != grid {H_p}x{W_p}."
        )

        # Perform 2D rFFT in float32 for precision under AMP FP16
        z_grid = z_patch.float().view(B, H_p, W_p, self.rank)
        z_freq = torch.fft.rfft2(z_grid, dim=(1, 2), norm="ortho")  # (B, H_p, W_p//2+1, r)

        # Compute phase-preserving real-valued gate G in (0, 2)
        z_cls = z_prefix[:, 0, :].float()  # global [CLS] token (B, r)
        band_logits = self.compute_band_logits(z_cls)  # (B, K, r)

        # Project K bands onto 2D frequency plane via precomputed radial masks
        # masks: (K, H_p, W_f), band_logits: (B, K, r) -> gate_map: (B, H_p, W_f, r)
        gate_map = torch.einsum("khw,bkr->bhwr", self.radial_masks, band_logits)
        gate = 1.0 + torch.tanh(gate_map)  # strictly positive real in (0, 2)

        # Amplitude modulation (phase strictly preserved since gate is real & positive)
        z_freq_mod = z_freq * gate

        # Inverse 2D rFFT back to spatial domain
        z_spatial = torch.fft.irfft2(z_freq_mod, s=(H_p, W_p), dim=(1, 2), norm="ortho")
        z_patch_out = z_spatial.reshape(B, H_p * W_p, self.rank).to(orig_dtype)

        # Residual skip-concat for prefix tokens + Up-projection B
        z_out = torch.cat([z_prefix, z_patch_out], dim=1)
        return self.lora_B(z_out) * self.scaling

    @torch.no_grad()
    def get_static_band_gains(self) -> torch.Tensor:
        """Returns mean static gain 1 + tanh(w_k) per band of shape (K,)."""
        return (1.0 + torch.tanh(self.band_weights)).mean(dim=-1).cpu()


class AdaptedQKV(nn.Module):
    """
    Wraps timm's fused Attention.qkv Linear(d, 3*d) and applies PEFT adaptation
    to Query (slice 0:d) and Value (slice 2d:3d) projections.
    Supports: lora, dora, fourierft, sglora_static, sglora_dynamic.
    """

    def __init__(
        self,
        base_qkv: nn.Linear,
        method: str = "sglora_dynamic",
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        grid_size: Tuple[int, int] = (16, 16),
        num_prefix_tokens: int = 1,
        num_bands: int = 4,
        n_freqs: int = 1000,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.base_qkv = base_qkv
        self.method = method
        self.dim = base_qkv.in_features
        self.layer_idx = layer_idx

        # Freeze pre-trained QKV weights
        for p in self.base_qkv.parameters():
            p.requires_grad = False

        if method == "lora":
            self.q_branch = LoRABranch(self.dim, rank, alpha, dropout)
            self.v_branch = LoRABranch(self.dim, rank, alpha, dropout)
        elif method == "dora":
            self.q_branch = LoRABranch(self.dim, rank, alpha, dropout)
            self.v_branch = LoRABranch(self.dim, rank, alpha, dropout)
            # Magnitude vectors m initialized to row-wise L2 norms of frozen W_q and W_v
            w_q = base_qkv.weight[: self.dim, :].detach()
            w_v = base_qkv.weight[2 * self.dim :, :].detach()
            self.m_q = nn.Parameter(w_q.norm(p=2, dim=1))  # (d,)
            self.m_v = nn.Parameter(w_v.norm(p=2, dim=1))  # (d,)
        elif method == "fourierft":
            self.q_branch = FourierFTBranch(self.dim, n_freqs=n_freqs, seed=100 + layer_idx)
            self.v_branch = FourierFTBranch(self.dim, n_freqs=n_freqs, seed=500 + layer_idx)
        elif method in ("sglora_static", "sglora_dynamic"):
            is_dyn = method == "sglora_dynamic"
            self.q_branch = SGLoRABranch(
                self.dim, rank, alpha, dropout, grid_size, num_prefix_tokens, num_bands, dynamic=is_dyn
            )
            self.v_branch = SGLoRABranch(
                self.dim, rank, alpha, dropout, grid_size, num_prefix_tokens, num_bands, dynamic=is_dyn
            )
        else:
            raise ValueError(f"Unknown PEFT method: {method}")

    def _dora_proj(self, x: torch.Tensor, w0: torch.Tensor, b0: Optional[torch.Tensor],
                   branch: LoRABranch, mag: nn.Parameter) -> torch.Tensor:
        """Exact DoRA forward pass with detached directional norm (Liu et al., ICML 2024)."""
        delta_w = (branch.lora_B.weight @ branch.lora_A.weight) * branch.scaling
        w_eff = w0 + delta_w
        # Detach column/row norm for memory-efficient gradient as in DoRA Eq. (11)
        w_norm = w_eff.norm(p=2, dim=1, keepdim=True).detach().clamp_min(1e-6)
        w_scaled = (mag.unsqueeze(1) / w_norm) * w_eff
        return F.linear(x, w_scaled.to(x.dtype), b0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "dora":
            w = self.base_qkv.weight
            b = self.base_qkv.bias
            b_q = b[: self.dim] if b is not None else None
            b_k = b[self.dim : 2 * self.dim] if b is not None else None
            b_v = b[2 * self.dim :] if b is not None else None

            q = self._dora_proj(x, w[: self.dim], b_q, self.q_branch, self.m_q)
            k = F.linear(x, w[self.dim : 2 * self.dim], b_k)
            v = self._dora_proj(x, w[2 * self.dim :], b_v, self.v_branch, self.m_v)
            return torch.cat([q, k, v], dim=-1)

        # Standard additive path (LoRA, FourierFT, SG-LoRA)
        qkv = self.base_qkv(x)
        dq = self.q_branch(x)
        dv = self.v_branch(x)
        # Non-inplace slice addition for autograd safety
        q = qkv[..., : self.dim] + dq
        k = qkv[..., self.dim : 2 * self.dim]
        v = qkv[..., 2 * self.dim :] + dv
        return torch.cat([q, k, v], dim=-1)


def inject_peft(
    model: nn.Module,
    method: str = "sglora_dynamic",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    num_bands: int = 4,
    n_freqs: int = 1000,
) -> nn.Module:
    """
    Configures a timm VisionTransformer (e.g., vit_base_patch14_dinov2.lvd142m)
    for one of: linear_probe, full_ft, lora, dora, fourierft, sglora_static, sglora_dynamic.
    """
    if method == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
        return model

    # Freeze entire backbone first
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze classification head
    if hasattr(model, "head") and model.head is not None:
        for p in model.head.parameters():
            p.requires_grad = True

    if method == "linear_probe":
        return model

    # Determine ViT spatial grid and prefix tokens
    grid_size = getattr(model.patch_embed, "grid_size", (16, 16))
    num_prefix = getattr(model, "num_prefix_tokens", 1)

    for idx, blk in enumerate(model.blocks):
        blk.attn.qkv = AdaptedQKV(
            base_qkv=blk.attn.qkv,
            method=method,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            grid_size=grid_size,
            num_prefix_tokens=num_prefix,
            num_bands=num_bands,
            n_freqs=n_freqs,
            layer_idx=idx,
        )
    return model


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def extract_spectral_profile(model: nn.Module) -> Dict[str, List[List[float]]]:
    """Extracts learned static radial band gains across layers for interpretability plots."""
    q_gains, v_gains = [], []
    for blk in getattr(model, "blocks", []):
        qkv = blk.attn.qkv
        if isinstance(qkv, AdaptedQKV) and isinstance(qkv.q_branch, SGLoRABranch):
            q_gains.append(qkv.q_branch.get_static_band_gains().tolist())
            v_gains.append(qkv.v_branch.get_static_band_gains().tolist())
    return {"q_band_gains": q_gains, "v_band_gains": v_gains}
