import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import einsum

from monai.networks.layers.utils import get_norm_layer, get_dropout_layer
from monai.networks.layers.factories import Conv
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from diffusers.models.attention_processor import Attention


class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


def l2norm(t):
    return F.normalize(t, dim=-1)


def masked_mean(t, *, dim, mask=None):
    if mask is not None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        scale=8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities and masking

        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if mask is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_latents_mean_pooled=4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len=512,
        ff_mult=4
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n=num_latents_mean_pooled)
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if self.to_latents_from_mean_pooled_seq is not None:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents

class AttentionPooling(nn.Module):
    def __init__(self, num_channels, context_dim, head_dim=64):
        super().__init__()
        self.attention = Attention(
            context_dim,
            context_dim,
            dim_head=head_dim,
            out_dim=num_channels,
            out_bias=False,
        )

    def forward(self, context, context_mask=None):
        context_mask = context_mask.to(dtype=context.dtype)
        context = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        return context.squeeze(1)


class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.proj = nn.Linear(in_channels, out_channels * 2, bias=True)

    def forward(self, x):
        # x expected to be [B, C, *]
        # Workaround as layer norm can't currently be applied on arbitrary dimension: https://github.com/pytorch/pytorch/issues/71465
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2)  # -> [B, C, N] -> [B, N, C]
        x = self.norm(x)
        x, gate = self.proj(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        return x.transpose(1, 2).reshape(b, -1, *spatial)  # -> [B, C, N] -> [B, C, *]


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def compute_attention(q, k, v, num_heads, scale):
    q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> (b h) d n', h=num_heads), (q, k, v))  # [(BxHeads), Dim_per_head, N]

    attn = (torch.einsum('b d i, b d j -> b i j', q*scale, k*scale)).softmax(dim=-1)  # Matrix product = [(BxHeads), Dim_per_head, N] * [(BxHeads), Dim_per_head, N'] =[(BxHeads), N, N']

    out = torch.einsum('b i j, b d j-> b d i', attn, v)  # Matrix product: [(BxHeads), N, N'] * [(BxHeads), Dim_per_head, N'] = [(BxHeads), Dim_per_head, N] 
    out = rearrange(out, '(b h) d n-> b (h d) n', h=num_heads)  # ->  [B, (Heads x Dim_per_head), N]

    return out


class LinearTransformerNd(nn.Module):
    """ Combines multi-head self-attention and multi-head cross-attention.

        Multi-Head Self-Attention: 
        Similar to  multi-head self-attention (https://arxiv.org/abs/1706.03762) without Norm+MLP (compare Monai TransformerBlock)
        Proposed here: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
        Similar to: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/diffusionmodules/openaimodel.py#L278
        Similar to: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L80
        Similar to: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/dfbafee555bdae80b55d63a989073836bbfc257e/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L209 
        Similar to: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L150 

        CrossAttention:
            Proposed here: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L152
    """
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,  # WARNING: if out_channels != in_channels, skip connection is disabled
        num_heads=8,
        ch_per_head=32,  # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups': 32, "affine": True}),  # Or use LayerNorm but be aware of https://github.com/pytorch/pytorch/issues/71465 (=> GroupNorm with num_groups=1) 
        dropout=None,
        emb_dim=None,
    ):
        super().__init__()
        hid_channels = num_heads*ch_per_head
        self.num_heads = num_heads
        self.scale = ch_per_head**-0.25  # Should be 1/sqrt("queries and keys of dimension"), Note: additional sqrt needed as it follows OpenAI:  (q * scale) * (k * scale) instead of (q *k) * scale
        self.norm_x = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        emb_dim = in_channels if emb_dim is None else emb_dim

        Convolution = Conv["conv", spatial_dims]
        self.to_q = Convolution(in_channels, hid_channels, 1)
        self.to_k = Convolution(emb_dim, hid_channels, 1)
        self.to_v = Convolution(emb_dim, hid_channels, 1)

        self.to_out = nn.Sequential(
            zero_module(Convolution(hid_channels, out_channels, 1)),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # if no embedding is given, cross-attention defaults to self-attention

        # Normalize
        b, c, *spatial = x.shape
        x_n = self.norm_x(x)

        # Attention:  embedding (cross-attention) or x (self-attention)
        if embedding is None:
            embedding = x_n  # WARNING: This assumes that emb_dim==in_channels
        else:
            if embedding.ndim == 2:
                embedding = embedding.reshape(*embedding.shape[:2], *[1]*(x.ndim-2)) # [B, C*] -> [B, C*, *]
            # Why no normalization for embedding here?

        # Convolution 
        q = self.to_q(x_n)       # -> [B, (Heads x Dim_per_head), *]
        k = self.to_k(embedding) # -> [B, (Heads x Dim_per_head), *]
        v = self.to_v(embedding) # -> [B, (Heads x Dim_per_head), *]

        # Flatten
        q = q.reshape(b, c, -1)                  # -> [B, (Heads x Dim_per_head), N]
        k = k.reshape(*embedding.shape[:2], -1)  # -> [B, (Heads x Dim_per_head), N']
        v = v.reshape(*embedding.shape[:2], -1)  # -> [B, (Heads x Dim_per_head), N']

        # Apply attention
        out = compute_attention(q, k, v, self.num_heads, self.scale)

        out = out.reshape(*out.shape[:2], *spatial) # -> [B, (Heads x Dim_per_head), *]
        out = self.to_out(out)                      # -> [B, C', *]

        if x.shape == out.shape:
            out = x + out
        return out  # [B, C', *]


class LinearTransformer(nn.Module):
    """ See LinearTransformer, however this implementation is fixed to Conv1d/Linear"""
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,  # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads,
        ch_per_head=32,  # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups': 32, "affine": True}),
        dropout=None,
        emb_dim=None
    ):
        super().__init__()
        hid_channels = num_heads*ch_per_head
        self.num_heads = num_heads
        self.scale = ch_per_head**-0.25  # Should be 1/sqrt("queries and keys of dimension"), Note: additional sqrt needed as it follows OpenAI:  (q * scale) * (k * scale) instead of (q *k) * scale 

        self.norm_x = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        emb_dim = in_channels if emb_dim is None else emb_dim

        # Note: Conv1d and Linear are interchangeable but order of input changes [B, C, N] <-> [B, N, C]
        self.to_q = nn.Conv1d(in_channels, hid_channels, 1)
        self.to_k = nn.Conv1d(emb_dim, hid_channels, 1)
        self.to_v = nn.Conv1d(emb_dim, hid_channels, 1)
        # self.to_qkv = nn.Conv1d(emb_dim, hid_channels*3, 1)

        self.to_out = nn.Sequential(
            zero_module(nn.Conv1d(hid_channels, out_channels, 1)),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        )

    def forward(self, x, embedding=None, condition=None, condition_mask=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # if no embedding is given, cross-attention defaults to self-attention

        # Normalize
        b, c, *spatial = x.shape

        # print('x:', x.shape)
        x_n = self.norm_x(x)

        # print('x_n:', x_n.shape)

        # Attention:  embedding (cross-attention) or x (self-attention)
        if embedding is None:
            embedding = x_n  # WARNING: This assumes that emb_dim==in_channels
        else:
            if embedding.ndim == 2:
                embedding = embedding.reshape(*embedding.shape[:2], *[1]*(x.ndim-2))  # [B, C*] -> [B, C*, *]
            # Why no normalization for embedding here?
        # print('emb:', embedding.shape)
        # Flatten
        x_n = x_n.reshape(b, c, -1)                              # [B, C,  *] -> [B, C,  N]

        # print('x_n new shape:', x_n.shape)
        embedding = embedding.reshape(*embedding.shape[:2], -1)  # [B, C*, *] -> [B, C*, N']
        # print('new emb shape:', embedding.shape)
        # Convolution 
        q = self.to_q(x_n)       # -> [B, (Heads x Dim_per_head), N]
        # print('q:', q.shape)
        k = self.to_k(embedding)  # -> [B, (Heads x Dim_per_head), N']
        # print('k:', k.shape)
        v = self.to_v(embedding)  # -> [B, (Heads x Dim_per_head), N']
        # print('v:', v.shape)
        # qkv = self.to_qkv(x_n)
        # q,k,v = qkv.split(qkv.shape[1]//3, dim=1)

        # Apply attention
        out = compute_attention(q, k, v, self.num_heads, self.scale)

        out = self.to_out(out)                      # -> [B, C', N]
        out = out.reshape(*out.shape[:2], *spatial)  # -> [B, C', *]

        if x.shape == out.shape:
            out = x + out
        return out  # [B, C', *]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,  # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads,
        ch_per_head=32,
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        dropout=None,
        emb_dim=None
    ):
        super().__init__()
        self.self_atn = LinearTransformer(spatial_dims, in_channels, in_channels, num_heads, ch_per_head, norm_name, dropout, None)
        if emb_dim is not None:
            self.cros_atn = LinearTransformer(spatial_dims, in_channels, in_channels, num_heads, ch_per_head, norm_name, dropout, emb_dim)
        self.proj_out = nn.Sequential(
            GEGLU(in_channels, in_channels * 4),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims),
            Conv["conv", spatial_dims](in_channels * 4, out_channels, 1, bias=True)
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        x = self.self_atn(x)
        if embedding is not None:
            x = self.cros_atn(x, embedding=embedding)
        out = self.proj_out(x)
        if out.shape[1] == x.shape[1]:
            return out + x
        return x


class SpatialTransformer(nn.Module):
    """ Proposed here: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L218
        Unrelated to: https://arxiv.org/abs/1506.02025  
    """
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,  # WARNING: if out_channels != in_channels, skip connection is disabled
        num_heads,
        ch_per_head=32,  # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        dropout=None,
        emb_dim=None,
        depth=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        conv_class = Conv["conv", spatial_dims]
        hid_channels = num_heads*ch_per_head

        self.proj_in = conv_class(
            in_channels,
            hid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(spatial_dims, hid_channels, hid_channels, num_heads, ch_per_head, norm_name, dropout=dropout, emb_dim=emb_dim)
            for _ in range(depth)]
        )

        self.proj_out = conv_class( # Note: zero_module is used in original code
            hid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # Note: if no embedding is given, cross-attention is disabled
        h = self.norm(x)
        h = self.proj_in(h)

        for block in self.transformer_blocks:
            h = block(h, embedding=embedding)

        h = self.proj_out(h)  # -> [B, C'', *]
        if h.shape == x.shape:
            return h + x
        return h


class ConditionalGroupNorm(nn.Module):
    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, 2 * normalized_shape))
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)

        for _ in range(len(x.shape[2:])):
            context = context.unsqueeze(-1)

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.0) + shift
        return x


class ConditionalLinearTransformer(nn.Module):
    def __init__(self, num_channels, time_embed_dim, context_dim=None, num_heads=8, norm_groups=32, head_dim=64, dropout=0, expansion_ratio=4):
        super().__init__()
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(
            num_channels,
            context_dim or num_channels,
            heads=num_heads,
            dim_head=head_dim,
            out_dim=num_channels,
            out_bias=False,
            dropout=dropout
        )

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, emb, context=None, context_mask=None):
        height, width = x.shape[-2:]
        out = self.in_norm(x, emb)
        out = out.reshape(x.shape[0], -1, height * width).permute(0, 2, 1)
        context = context if context is not None else out
        if context_mask is not None:
            context_mask = context_mask.to(dtype=context.dtype)

        out = self.attention(out, context, context_mask)
        out = out.permute(0, 2, 1).unsqueeze(-1).reshape(out.shape[0], -1, height, width)
        x = x + out

        out = self.out_norm(x, emb)
        out = self.feed_forward(out)
        x = x + out
        return x


class MedfusionAttention(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        num_heads=8,
        ch_per_head=32,  # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups': 32, "affine": True}),
        dropout=0,
        emb_dim=None,
        depth=1,
        attention_type='linear'
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        if attention_type == 'spatial':
            self.attention = SpatialTransformer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                ch_per_head=ch_per_head,
                depth=depth,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=emb_dim 
            )
        elif attention_type == 'linear':
            self.attention = LinearTransformer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                ch_per_head=ch_per_head,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=emb_dim
            )
        elif attention_type == 'diffusers':
            self.attention = ConditionalLinearTransformer(
                num_channels=in_channels,
                time_embed_dim=emb_dim,
                context_dim=emb_dim,
                num_heads=num_heads,
                head_dim=ch_per_head,
                dropout=dropout
            )

    def forward(self, x, emb=None, condition=None, condition_mask=None):
        if hasattr(self, 'attention'):
            return self.attention(x, emb, condition, condition_mask)
        else:
            return x
