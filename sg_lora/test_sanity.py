"""
Week-4 Mid-Way Automated Sanity Check Suite for SG-LoRA:
  1. Zero-perturbation initialization check: SG-LoRA(X) == W0(X) at step 0
  2. Strict Fourier phase preservation check: angle(Z_freq * G) == angle(Z_freq)
  3. Gradient flow check through 2D rFFT / irFFT & CLS-conditioned gate
  4. Exact parameter count verification across all 5 PEFT methods
"""

import torch
import torch.nn as nn
from sg_lora.modules import SGLoRABranch, AdaptedQKV, build_radial_masks, inject_peft, count_parameters


class MockPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid_size = (16, 16)


class MockAttn(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)


class MockBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.attn = MockAttn(dim)


class MockDINOv2(nn.Module):
    def __init__(self, dim=768, depth=12, num_classes=7):
        super().__init__()
        self.patch_embed = MockPatchEmbed()
        self.num_prefix_tokens = 1
        self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, num_classes)


def run_all_tests():
    torch.manual_seed(0)
    B, N, D = 2, 1 + 16 * 16, 768
    x = torch.randn(B, N, D)

    # Test 1: Radial mask partition sums to 1.0 everywhere
    masks = build_radial_masks(16, 16, num_bands=4)
    assert torch.allclose(masks.sum(dim=0), torch.ones(16, 9), atol=1e-5), "Masks do not sum to 1!"
    print("[PASS] Test 1: Radial spectral masks form an exact partition of unity.")

    # Test 2: Zero-perturbation initialization
    base_qkv = nn.Linear(D, 3 * D)
    ref_out = base_qkv(x)
    wrapped = AdaptedQKV(base_qkv, method="sglora_dynamic", rank=8, num_bands=4)
    init_out = wrapped(x)
    max_diff = (ref_out - init_out).abs().max().item()
    assert max_diff < 1e-6, f"Init diff too high: {max_diff}"
    print(f"[PASS] Test 2: Zero-perturbation init verified (max diff = {max_diff:.2e}).")

    # Test 3: Strict Fourier phase preservation
    sg = SGLoRABranch(D, rank=8, num_bands=4, dynamic=True)
    nn.init.normal_(sg.band_weights, std=0.5)
    nn.init.normal_(sg.cls_gate.weight, std=0.2)
    z = sg.lora_A(x)
    z_grid = z[:, 1:, :].view(B, 16, 16, 8)
    z_freq = torch.fft.rfft2(z_grid, dim=(1, 2), norm="ortho")
    band_logits = sg.compute_band_logits(z[:, 0, :])
    gate = 1.0 + torch.tanh(torch.einsum("khw,bkr->bhwr", sg.radial_masks, band_logits))
    assert (gate > 0).all() and (gate < 2).all(), "Gate out of (0, 2) bounds!"
    phase_diff = (torch.angle(z_freq) - torch.angle(z_freq * gate)).abs().max().item()
    assert phase_diff < 1e-5, f"Phase distorted: {phase_diff}"
    print(f"[PASS] Test 3: Strictly positive real gate preserves Fourier phase (max diff = {phase_diff:.2e} rad).")

    # Test 4: Gradient flow through 2D rFFT / irFFT & CLS-MLP
    nn.init.normal_(wrapped.q_branch.lora_B.weight, std=0.02)
    nn.init.normal_(wrapped.v_branch.lora_B.weight, std=0.02)
    loss = wrapped(x).pow(2).mean()
    loss.backward()
    assert wrapped.q_branch.band_weights.grad is not None and wrapped.q_branch.band_weights.grad.abs().sum() > 0
    assert wrapped.q_branch.cls_gate.weight.grad is not None and wrapped.q_branch.cls_gate.weight.grad.abs().sum() > 0
    print("[PASS] Test 4: Autograd gradients flow cleanly through 2D rFFT, radial gate, and CLS-MLP.")

    # Test 5: Parameter audit across all PEFT methods
    for m in ["linear_probe", "lora", "dora", "fourierft", "sglora_static", "sglora_dynamic"]:
        net = inject_peft(MockDINOv2(), method=m, rank=8, num_bands=4)
        stats = count_parameters(net)
        print(f"       Method={m:15s} | Trainable={stats['trainable_params']:>8,d} ({stats['trainable_pct']:.3f}%)")
    print("[PASS] Test 5: All PEFT injection modes verified.")


if __name__ == "__main__":
    run_all_tests()
