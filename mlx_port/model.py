"""A1 Llama port. Parameter paths and matrix layouts match modeling.py."""
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint as checkpoint_module
from mlx.utils import tree_flatten, tree_unflatten

VARIANTS = ("reference", "no_prenorm", "no_residuals", "no_mlp", "no_attention", "no_rope", "mixed_residual")


@dataclass(frozen=True)
class Config:
    name: str = "d8"
    vocab_size: int = 4096
    context_length: int = 1024
    hidden_size: int = 512
    intermediate_size: int = 1792
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072

    @classmethod
    def tiny(cls):
        return cls(name="tiny", vocab_size=128, context_length=16,
                   hidden_size=64, intermediate_size=224, num_hidden_layers=2,
                   num_attention_heads=2, num_key_value_heads=2, head_dim=32)


def depth_model_config(depth, vocab_size=4096, context_length=1024):
    if not 4 <= depth <= 18:
        raise ValueError("depth must be in [4,18]")
    width = 64 * depth
    return Config(name=f"d{depth}", vocab_size=vocab_size, context_length=context_length,
                  hidden_size=width, intermediate_size=int(width * 3.5),
                  num_hidden_layers=depth, num_attention_heads=depth,
                  num_key_value_heads=depth)


class Linear(nn.Module):
    def __init__(self, inputs, outputs, dtype):
        super().__init__()
        self.weight = mx.zeros((outputs, inputs))
        self._dtype = dtype

    def __call__(self, x):
        return x.astype(self._dtype) @ self.weight.astype(self._dtype).T


class RMSNorm(nn.Module):
    def __init__(self, dims, precision):
        super().__init__()
        self.weight = mx.ones((dims,))
        self._precision = precision

    def __call__(self, x):
        # Reference normalizes in fp32, casts back, then multiplies by weight.
        if self._precision == "bf16":
            return mx.fast.rms_norm(x, self.weight.astype(x.dtype), 1e-5)
        if x.dtype == mx.float32:
            return mx.fast.rms_norm(x, self.weight, 1e-5)
        return mx.fast.rms_norm(x, None, 1e-5) * self.weight


class Attention(nn.Module):
    def __init__(self, c, precision, dropout=0.0, variant="reference"):
        super().__init__()
        self._c = c
        self._variant = variant
        self.prob_dropout = nn.Dropout(dropout)
        self._dtype = mx.float32 if precision == "fp32" else mx.bfloat16
        self.q_proj = Linear(c.hidden_size, c.num_attention_heads*c.head_dim, self._dtype)
        self.k_proj = Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, self._dtype)
        self.v_proj = Linear(c.hidden_size, c.num_key_value_heads*c.head_dim, self._dtype)
        self.o_proj = Linear(c.hidden_size, c.hidden_size, self._dtype)
        self.q_norm = RMSNorm(c.head_dim, precision)
        self.k_norm = RMSNorm(c.head_dim, precision)

    def __call__(self, x, attention_mask=None, position_ids=None):
        c = self._c
        b, t, _ = x.shape
        def project(layer, heads):
            return layer(x).reshape(b, t, heads, c.head_dim).transpose(0, 2, 1, 3)
        q, k, v = (project(self.q_proj, c.num_attention_heads),
                   project(self.k_proj, c.num_key_value_heads),
                   project(self.v_proj, c.num_key_value_heads))
        def rope(a):
            if self._variant == "no_rope":
                return a
            if position_ids is None:
                # False pairs first/second halves, exactly reference rotate_half.
                return mx.fast.rope(a, c.head_dim, traditional=False,
                                    base=500000., scale=1., offset=0)
            freqs = position_ids.astype(mx.float32)[..., None] * (
                500000. ** (-mx.arange(0, c.head_dim, 2, dtype=mx.float32)/c.head_dim))
            emb = mx.concatenate([freqs, freqs], axis=-1)[:, None]
            cs, sn = mx.cos(emb).astype(a.dtype), mx.sin(emb).astype(a.dtype)
            half = c.head_dim//2
            rot = mx.concatenate([-a[..., half:], a[..., :half]], axis=-1)
            return a*cs + rot*sn
        # QK norm MUST be after RoPE.
        q = self.q_norm(rope(q)).astype(self._dtype)
        k = self.k_norm(rope(k)).astype(self._dtype)
        mask = "causal"
        if attention_mask is not None:
            causal = mx.arange(t)[:, None] >= mx.arange(t)[None, :]
            allowed = causal[None, None] & attention_mask[:, None, None, :].astype(mx.bool_)
            # Reference uses finite minimum, including fully masked query rows.
            mask = mx.where(allowed, 0., mx.finfo(q.dtype).min).astype(q.dtype)
        if self.training and self.prob_dropout._p_1 != 1:
            # MLX fused SDPA has no dropout argument. Apply inverted dropout
            # to softmax probabilities before V, exactly PyTorch SDPA semantics.
            scores = (q * c.head_dim**-0.5) @ k.swapaxes(-1, -2)
            if isinstance(mask, str):
                allowed = mx.arange(t)[:, None] >= mx.arange(t)[None, :]
                scores = mx.where(allowed, scores, mx.finfo(q.dtype).min)
            else:
                scores = scores + mask
            probabilities = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
            y = self.prob_dropout(probabilities) @ v
        else:
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=c.head_dim**-0.5, mask=mask)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(b, t, c.hidden_size))


class MLP(nn.Module):
    def __init__(self, c, precision):
        super().__init__()
        dt = mx.float32 if precision == "fp32" else mx.bfloat16
        self.gate_proj = Linear(c.hidden_size, c.intermediate_size, dt)
        self.up_proj = Linear(c.hidden_size, c.intermediate_size, dt)
        self.down_proj = Linear(c.intermediate_size, c.hidden_size, dt)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x))*self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c, precision, dropout=0.0, variant="reference"):
        super().__init__()
        self._variant = variant
        self.self_attn = Attention(c, precision, dropout, variant)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp = MLP(c, precision)
        self.input_layernorm = RMSNorm(c.hidden_size, precision)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, precision)

    def __call__(self, x, attention_mask=None, position_ids=None, previous=None):
        a = mx.zeros_like(x) if self._variant == "no_attention" else self.attention_dropout(
            self.self_attn(x if self._variant == "no_prenorm" else self.input_layernorm(x),
                           attention_mask, position_ids))
        h = a if self._variant == "no_residuals" else x + a
        m = mx.zeros_like(h) if self._variant == "no_mlp" else self.mlp_dropout(
            self.mlp(h if self._variant == "no_prenorm" else self.post_attention_layernorm(h)))
        if self._variant == "no_residuals":
            return m
        if self._variant == "mixed_residual" and previous is not None:
            return 0.5*x + 0.5*previous + a + m
        return h + m


class Backbone(nn.Module):
    def __init__(self, c, precision, checkpoint, dropout=0.0, variant="reference"):
        super().__init__()
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = [Layer(c, precision, dropout, variant) for _ in range(c.num_hidden_layers)]
        self.norm = RMSNorm(c.hidden_size, precision)
        self._precision = precision
        self._checkpoint = checkpoint
        self._variant = variant

    def __call__(self, ids, attention_mask=None, position_ids=None):
        x = self.embed_tokens(ids)
        if self._precision == "bf16":
            x = x.astype(mx.bfloat16)
        previous = None
        for layer in self.layers:
            fn = checkpoint_module(layer) if self._checkpoint else layer
            old = x
            if self._variant == "mixed_residual":
                x = fn(x, attention_mask, position_ids, previous)
            else:
                x = fn(x, attention_mask, position_ids)
            previous = old
        return self.norm(x)


class Model(nn.Module):
    def __init__(self, config=Config(), precision="fp32", checkpoint=False, dropout=0.0, variant="reference"):
        super().__init__()
        if precision not in ("fp32", "mp", "bf16"):
            raise ValueError(precision)
        self._config = config
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self._variant = variant
        if checkpoint and dropout:
            raise ValueError("Dropout with activation checkpointing is not supported (RNG replay)")
        self.model = Backbone(config, precision, checkpoint, dropout, variant)
        self.lm_head = Linear(config.hidden_size, config.vocab_size,
                              mx.float32 if precision == "fp32" else mx.bfloat16)

    def hidden(self, ids, attention_mask=None, position_ids=None):
        if ids.ndim != 2 or ids.shape[1] > self._config.max_position_embeddings:
            raise ValueError("Expected [batch, sequence] within max_position_embeddings")
        return self.model(ids, attention_mask, position_ids)

    def __call__(self, ids, attention_mask=None, position_ids=None):
        return self.lm_head(self.hidden(ids, attention_mask, position_ids))

    def initialize(self, seed=42):
        # Same truncated distribution and shared *initial* base as reference;
        # RNG algorithms differ, so parity loads actual reference weights.
        mx.random.seed(seed)
        weights = {}
        for name, p in tree_flatten(self.parameters()):
            if p.ndim == 1:
                weights[name] = mx.ones_like(p)
            elif name not in ("model.embed_tokens.weight", "lm_head.weight"):
                weights[name] = mx.random.truncated_normal(-3., 3., p.shape)/math.sqrt(p.shape[1])
        c = self._config
        base = mx.random.truncated_normal(-3., 3., (c.vocab_size, c.hidden_size))
        weights["model.embed_tokens.weight"] = base/c.hidden_size
        weights["lm_head.weight"] = base/math.sqrt(c.hidden_size)
        self.update(tree_unflatten(list(weights.items())))

    def load_reference(self, state):
        self.load_weights([(k, mx.array(v.detach().cpu().float().numpy())) for k, v in state.items()], strict=True)

    def unused_parameter(self, name):
        """Static bypasses: MLX gives zeros; torch gives None and skips AdamW."""
        if not name.startswith("model.layers."):
            return False
        return ((self._variant == "no_prenorm" and
                 (".input_layernorm." in name or ".post_attention_layernorm." in name)) or
                (self._variant == "no_mlp" and
                 (".mlp." in name or ".post_attention_layernorm." in name)) or
                (self._variant == "no_attention" and
                 (".self_attn." in name or ".input_layernorm." in name)))


def loss(model, ids, chunk_size=0):
    """Mean over B*(T-1), chunking the head BEFORE allocating vocabulary logits.

    Checkpoint each chunk to avoid retaining all its logits for backward.
    """
    h = model.hidden(ids)[:, :-1].reshape(-1, model._config.hidden_size)
    labels = ids[:, 1:].reshape(-1)
    def chunk_ce(x, targets, weight):
        targets = mx.stop_gradient(targets)
        logits = x.astype(model.lm_head._dtype) @ weight.astype(model.lm_head._dtype).T
        return mx.sum(mx.fast.cross_entropy(logits.astype(mx.float32), targets))
    if chunk_size:
        ce = mx.checkpoint(chunk_ce)
        total = mx.array(0., mx.float32)
        for start in range(0, labels.size, chunk_size):
            total = total + ce(h[start:start+chunk_size], labels[start:start+chunk_size], model.lm_head.weight)
        return total/labels.size
    return chunk_ce(h, labels, model.lm_head.weight)/labels.size
