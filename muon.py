"""
Muon optimizer - MomentUm Orthogonalized by Newton-Schulz
Based on Keller Jordan's implementation for modded-nanogpt.

Key idea: Apply Newton-Schulz orthogonalization to momentum updates
for 2D (matrix) parameters. Use AdamW for everything else
(embeddings, biases, layernorm).
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def newton_schulz_orthogonalize(G: Tensor, steps: int = 5) -> Tensor:
    """
    Orthogonalize G via Newton-Schulz iterations.
    Computes approx. U from G = U * S * V^T (SVD).
    Works on 2D tensors. For non-square, pads/transposes as needed.
    """
    assert G.ndim == 2
    a, b = G.shape
    transposed = False
    if a < b:
        G = G.T
        transposed = True

    # Normalize
    G = G / (G.norm() + 1e-7)

    # Newton-Schulz iteration: X_{k+1} = (3*X_k - X_k^3) / 2
    # Coefficients for a degree-5 polynomial (faster convergence)
    for _ in range(steps):
        A = G @ G.T
        G = (3.0 * G - A @ G) / 2.0

    if transposed:
        G = G.T
    return G


class Muon(Optimizer):
    """
    Muon optimizer for hidden layer matrix parameters.
    
    Uses AdamW for:
      - Embedding layers (wte, wpe)
      - LM head
      - LayerNorm / bias parameters
    
    Uses Muon (momentum + orthogonalization) for:
      - All other 2D weight matrices (attention, MLP)

    Args:
        muon_params:  iterable of 2D matrix parameters for Muon
        lr:           learning rate for Muon (default: 0.02)
        momentum:     momentum for Muon (default: 0.95)
        ns_steps:     Newton-Schulz iteration steps (default: 5)
        adamw_params: iterable of remaining parameters for AdamW
        adamw_lr:     learning rate for AdamW (default: 3e-4)
        adamw_betas:  betas for AdamW (default: (0.9, 0.95))
        adamw_wd:     weight decay for AdamW (default: 0.1)
    """

    def __init__(
        self,
        muon_params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_lr: float = 3e-4,
        adamw_betas=(0.9, 0.95),
        adamw_wd: float = 0.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_wd=adamw_wd,
        )
        params = list(muon_params)
        if adamw_params is not None:
            params += list(adamw_params)

        super().__init__(params, defaults)

        # Tag which params use Muon vs AdamW
        self.muon_params = set(id(p) for p in muon_params)
        if adamw_params is not None:
            self.adamw_params = set(id(p) for p in adamw_params)
        else:
            self.adamw_params = set()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                pid = id(p)
                state = self.state[p]
                g = p.grad

                if pid in self.muon_params and g.ndim == 2:
                    # --- Muon update for 2D matrix params ---
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["step"] = 0

                    state["step"] += 1
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)

                    # Orthogonalize the momentum buffer
                    g_orth = newton_schulz_orthogonalize(buf, steps=group["ns_steps"])

                    # Scale update to match RMS of gradient
                    scale = max(g.shape) ** 0.5
                    p.add_(g_orth, alpha=-group["lr"] * scale)

                else:
                    # --- AdamW update for everything else ---
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["step"] = 0

                    state["step"] += 1
                    beta1, beta2 = group["adamw_betas"]
                    eps = 1e-8
                    t = state["step"]

                    state["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bias_c1 = 1 - beta1 ** t
                    bias_c2 = 1 - beta2 ** t
                    step_size = group["adamw_lr"] / bias_c1
                    denom = (state["exp_avg_sq"].sqrt() / (bias_c2 ** 0.5)).add_(eps)

                    # Weight decay
                    p.mul_(1 - group["adamw_lr"] * group["adamw_wd"])
                    p.addcdiv_(state["exp_avg"], denom, value=-step_size)

        return loss