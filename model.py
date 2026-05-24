"""
GPT Language Model with RoPE, SwiGLU, and weight tying.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ── RoPE helpers ─────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, seq_len: int, device, base: float = 10000.0):
    """Returns (cos, sin) each of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, theta)          # (T, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)      # (T, head_dim)
    return freqs.cos(), freqs.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, nh, T, hs);  cos, sin: (T, hs)."""
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, hs)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ── Attention ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ── SwiGLU MLP ───────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """SwiGLU: uses two gates instead of one. Slightly smaller hidden dim to
    keep param count comparable to the original 4x MLP."""

    def __init__(self, config):
        super().__init__()
        # 8/3 * n_embd is the standard SwiGLU hidden size
        hidden = int(config.n_embd * 8 / 3)
        # round to nearest multiple of 64 for efficiency
        hidden = (hidden + 63) // 64 * 64

        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: silu(gate) * up
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int  = 1024
    vocab_size: int  = 50304
    n_layer:    int  = 12
    n_head:     int  = 12
    n_embd:     int  = 768
    dropout:    float = 0.0
    bias:       bool  = False


# ── GPT Model ─────────────────────────────────────────────────────────────────

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            # NO wpe — RoPE handles positional info
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying — lm_head shares weights with token embedding
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        # lm_head and wte are tied, so only count once
        if non_embedding:
            n_params -= self.lm_head.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Precompute RoPE frequencies for this sequence length
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = precompute_rope_freqs(head_dim, t, device)

        tok_emb = self.transformer.wte(idx)   # (b, t, n_embd)
        x = self.transformer.drop(tok_emb)    # no positional embedding added — RoPE handles it

        for block in self.transformer.h:
            x = block(x, cos, sin)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # No wpe to crop anymore — nothing else needed for RoPE

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # kept for compatibility but RoPE models can't load GPT-2 weights directly
        raise NotImplementedError("from_pretrained not supported with RoPE model")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params   = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **(dict(fused=True) if use_fused else dict())
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token  = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_achieved   = flops_per_fwdbwd * fwdbwd_per_iter / dt
        flops_promised   = 312e12  # A100 bf16 peak
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    def clean_state_dict(sd):
        return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                for k, v in sd.items()}

    model_args = checkpoint["model_args"]
    config = GPTConfig(**model_args)
    model = GPT(config)
    model.load_state_dict(clean_state_dict(checkpoint["model"]))
    model.to(device)
    model.eval()

    class WrappedModel(nn.Module):
        def __init__(self, m): super().__init__(); self.model = m
        def forward(self, input_ids):
            logits, _ = self.model(input_ids)
            return logits[:, :, :50257]

    return WrappedModel(model)