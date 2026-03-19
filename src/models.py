import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models
import math
import timm

class LayerNorm2d(nn.Module):
    def __init__(self, ch: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias   = nn.Parameter(torch.zeros(ch))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.layer_norm expects the normalised dims at the end → permute to NHWC,
        # normalise over C, then permute back. The fused kernel avoids the manual
        # mean/var ops and is safe for fp16/bf16.
        x = x.permute(0, 2, 3, 1)                          # NCHW → NHWC
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()          # NHWC → NCHW

class ViT_Encoder(nn.Module):
    def __init__(self, target_spatial_ch, label_emb_size, side_pixels):
        super().__init__()

        self.vit = timm.create_model(
            'vit_small_patch14_dinov2', 
            pretrained=True,
            num_classes=0,
            img_size=side_pixels
        )

        for name, p in self.vit.named_parameters():
            if any(name.startswith(x) for x in
                   ['patch_embed', 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3']):
                p.requires_grad = False

        vit_dim = self.vit.num_features

        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, target_spatial_ch),
            nn.GELU(),
        )

        self.global_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, label_emb_size),
            nn.GELU(),
        )

        self.spatial_norm2d = LayerNorm2d(target_spatial_ch)

        self.null_global = nn.Parameter(torch.randn(label_emb_size))
        self.null_spatial = nn.Parameter(torch.randn(1, target_spatial_ch, 1, 1))

    def forward(self, x, drop_mask=None):
        B = x.shape[0]

        tokens = self.vit.forward_features(x)

        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]

        h = w = int(patch_tokens.shape[1] ** 0.5)
        
        sp_feat = self.spatial_proj(patch_tokens)
        sp_feat = sp_feat.permute(0, 2, 1).reshape(B, -1, h, w)
        sp_feat = self.spatial_norm2d(sp_feat)

        global_feat = self.global_proj(cls_token)

        if drop_mask is not None:
            mask_1d = drop_mask.view(B, 1).float()
            mask_4d = drop_mask.view(B, 1, 1, 1).float()

            global_feat = global_feat * (1.0 - mask_1d) + self.null_global.unsqueeze(0) * mask_1d
            sp_feat = sp_feat * (1.0 - mask_4d) + self.null_spatial * mask_4d

        return sp_feat, global_feat

class ResNet_Encoder(nn.Module):

    _RESNET_CHS = [256, 128, 64, 64]

    def __init__(self, label_emb_size, decoder_channels, condition_ch=3, denses_arr=None):
        super().__init__()

        n_scales = len(decoder_channels)
        assert n_scales <= len(self._RESNET_CHS), (
            f"Requested {n_scales} encoder scales but only "
            f"{len(self._RESNET_CHS)} ResNet levels are available."
        )

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if condition_ch == 1:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)

        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        for p in self.stem.parameters():
            p.requires_grad = False

        src_chs = self._RESNET_CHS[:n_scales]
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(src_ch, dec_ch, kernel_size=1),
                nn.GroupNorm(min(8, dec_ch), dec_ch),
                nn.GELU(),
            )
            for src_ch, dec_ch in zip(src_chs, decoder_channels)
        ])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        if not denses_arr:
            self.fc = nn.Sequential(nn.Linear(256, label_emb_size), nn.GELU())
        else:
            layers_list = []
            in_dim = 256
            for h in denses_arr:
                layers_list += [nn.Linear(in_dim, h), nn.GELU()]
                in_dim = h
            layers_list.append(nn.Linear(in_dim, label_emb_size))
            self.fc = nn.Sequential(*layers_list)

        self.n_scales = n_scales

    def forward(self, x, drop_mask=None):
        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1) 
        s3 = self.layer3(s2)

        raw_feats = [s3, s2, s1, s0][:self.n_scales]

        multi_scale_feats = [proj(f) for proj, f in zip(self.scale_projs, raw_feats)]

        if drop_mask is not None:
            multi_scale_feats = [
                f.masked_fill(drop_mask[:, None, None, None], 0.0)
                for f in multi_scale_feats
            ]

        global_feat = self.fc(self.flatten(self.global_pool(s3)))
        if drop_mask is not None:
            global_feat = global_feat.masked_fill(drop_mask[:, None], 0.0)

        return multi_scale_feats, global_feat

class ResidualBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(emb_dim, ch * 2)
        self.norm2 = LayerNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _impl(self, x, emb):
        h = self.conv1(F.gelu(self.norm1(x)))
        
        h_norm = self.norm2(h)
        gamma, beta = torch.chunk(self.film(emb)[:, :, None, None], 2, dim=1)
        h_mod = h_norm * (1 + gamma) + beta
        h_out = self.conv2(F.gelu(h_mod))
        
        return x + h_out

    def forward(self, x, emb=None):
        if self.training:
            return checkpoint(self._impl, x, emb, use_reentrant=False)
        return self._impl(x, emb)

class Image_Encoder(nn.Module):
    def __init__(self, filters_arr, denses_arr, label_emb_size, decoder_channels=None, in_channels=3, 
                 n_channels_group=8, side_pixels=128):
        super().__init__()
        self.downs = nn.ModuleList()
        
        for i in range(len(filters_arr)):
            in_ch = in_channels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            
            self.downs.append(nn.ModuleDict({
                'conv': nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                'residual': ResidualBlock(out_ch, emb_dim=out_ch)
            }))
            
        self.act_gelu = nn.GELU()
        self.final_norm = nn.GroupNorm(n_channels_group, filters_arr[-1])

        if decoder_channels is not None:
            self.n_scales = len(decoder_channels)
            src_chs = list(reversed(filters_arr))[:self.n_scales]
            
            self.scale_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(src_ch, dec_ch, kernel_size=1),
                    nn.GroupNorm(min(n_channels_group, dec_ch), dec_ch),
                    nn.GELU(),
                )
                for src_ch, dec_ch in zip(src_chs, decoder_channels)
            ])
        else:
            self.n_scales = 0
            self.scale_projs = None

        resulting_side_pixels = side_pixels // (2 ** len(filters_arr))
        self.flatten = nn.Flatten()
        self.linears = nn.ModuleList()
        for i in range(len(denses_arr)):
            in_dim = filters_arr[-1] * (resulting_side_pixels**2) if i == 0 else denses_arr[i-1]
            self.linears.append(nn.Linear(in_dim, denses_arr[i]))
            
        last_in_dim = denses_arr[-1] if len(denses_arr) > 0 else filters_arr[-1] * (resulting_side_pixels**2)
        self.last = nn.Linear(last_in_dim, label_emb_size)

    def forward(self, x, drop_mask=None):
        raw_feats = []
        for down in self.downs:
            x = down['conv'](x)
            x = down['residual'](x)
            raw_feats.append(x)
        
        feat = self.act_gelu(self.final_norm(x))

        if self.scale_projs is not None:
            reversed_feats = list(reversed(raw_feats))[:self.n_scales]
            multi_scale_feats = [proj(f) for proj, f in zip(self.scale_projs, reversed_feats)]
            
            if drop_mask is not None:
                multi_scale_feats = [
                    f.masked_fill(drop_mask[:, None, None, None], 0.0)
                    for f in multi_scale_feats
                ]
        else:
            multi_scale_feats = [feat]
            if drop_mask is not None:
                multi_scale_feats = [f.masked_fill(drop_mask[:, None, None, None], 0.0) for f in multi_scale_feats]
        
        global_feat = self.flatten(feat)
        for linear in self.linears:
            global_feat = self.act_gelu(linear(global_feat))
        global_feat = self.last(global_feat)
        
        if drop_mask is not None:
            global_feat = global_feat.masked_fill(drop_mask[:, None], 0.0)

        return multi_scale_feats, global_feat
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, ch, n_channels_group=8):
        super().__init__()
        self.gn = nn.GroupNorm(n_channels_group, ch)
        self.k = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
        self.q = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
        self.v = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
        self.final = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)

        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

        self.act_softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        b, c, h, w = x.shape

        x_gn = self.gn(x)

        k = self.k(x_gn).view(b, c, -1)
        q = self.q(x_gn).view(b, c, -1)
        v = self.v(x_gn).view(b, c, -1)

        attn = torch.bmm(q.transpose(1,2), k) * (c**(-1/2))
        attn = self.act_softmax(attn)

        attn = torch.bmm(v, attn.transpose(1,2))
        attn = attn.view(b, c, h, w)

        return self.final(attn) + x
    
def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)

def get_2d_sincos_pos_embed(embed_dim, h, w, device):

    assert embed_dim % 4 == 0, "Embedding dimension must be divisible by 4 for 2D sin-cos pos embedding"
    half_dim = embed_dim // 2
    
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
    
    emb = torch.arange(half_dim // 2, dtype=torch.float32, device=device)
    emb = 10000.0 ** (2.0 * emb / half_dim)
    emb = 1.0 / emb
    
    emb_h = grid_h.flatten()[:, None] * emb[None, :] 
    emb_w = grid_w.flatten()[:, None] * emb[None, :] 
    
    pos_h = torch.cat([torch.sin(emb_h), torch.cos(emb_h)], dim=1)
    pos_w = torch.cat([torch.sin(emb_w), torch.cos(emb_w)], dim=1)
    
    pos = torch.cat([pos_h, pos_w], dim=1)
    
    return pos.view(h, w, embed_dim).permute(2, 0, 1).unsqueeze(0)


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, ch, n_channels_group=8):
        super().__init__()
        self.ch    = ch
        self.gn    = nn.GroupNorm(n_channels_group, ch)
        self.q     = nn.Conv2d(ch, ch, 1)
        self.k     = nn.Conv2d(ch, ch, 1)
        self.v     = nn.Conv2d(ch, ch, 1)
        self.final = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        self._pos_cache: dict = {}

    def _get_pos(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._pos_cache:
            pos = get_2d_sincos_pos_embed(self.ch, h, w, device).half()
            self._pos_cache[key] = pos
        return self._pos_cache[key]
    
    def _attn_impl(self, x, context, pos_x, pos_ctx):
        b, c, h, w = x.shape
        q = self.q(self.gn(x) + pos_x).flatten(2).transpose(1, 2)
        k = self.k(context + pos_ctx).flatten(2).transpose(1, 2)
        v = self.v(context).flatten(2).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        
        out = out.transpose(1, 2).view(b, c, h, w)
        return self.final(out) + x

    def forward(self, x, context):
        if self.training:
            pos_x = self._get_pos(x.shape[2], x.shape[3], x.device)
            pos_ctx = self._get_pos(context.shape[2], context.shape[3], context.device)
            return checkpoint(self._attn_impl, x, context, pos_x, pos_ctx, use_reentrant=False)
        return self._attn_impl(x, context, self._get_pos(*x.shape[2:], x.device), self._get_pos(*context.shape[2:], context.device))

class RelativePositionBias(nn.Module):
    def __init__(self, heads: int, max_h: int, max_w: int | None = None):
        super().__init__()
        max_w = max_w or max_h
        self.heads = heads
        self.max_h = max_h
        self.max_w = max_w
        table_h = 2 * max_h - 1
        table_w = 2 * max_w - 1
        self.rel = nn.Parameter(torch.randn(table_h * table_w, heads) * 0.02)
        self._idx_cache = None
        self._cache_shape: tuple[int, int] | None = None

    @torch.no_grad()
    def _build_index(self, h: int, w: int) -> torch.Tensor:
        device = self.rel.device
        gh = torch.arange(h, device=device)
        gw = torch.arange(w, device=device)
        # coords: (2, h*w)
        coords = torch.stack(
            torch.meshgrid(gh, gw, indexing="ij")
        ).flatten(1)

        rel = coords[:, :, None] - coords[:, None, :]

        rel[0] += self.max_h - 1
        rel[1] += self.max_w - 1
        idx = rel[0] * (2 * self.max_w - 1) + rel[1]
        return idx.long()

    def forward(self, h: int, w: int) -> torch.Tensor:
        if self._cache_shape != (h, w):
            self._idx_cache   = self._build_index(h, w)
            self._cache_shape = (h, w)

        return self.rel[self._idx_cache].permute(2, 0, 1).contiguous()
    
class MHAAttention(nn.Module):
    def __init__(self, ch, heads=4, head_dim=32, max_res=16):
        super().__init__()
        self.heads = heads
        self.scale = head_dim ** -0.5
        inner = heads * head_dim
        self.norm = LayerNorm2d(ch)
        self.to_q = nn.Conv2d(ch, inner, 1)
        self.to_k = nn.Conv2d(ch, inner, 1)
        self.to_v = nn.Conv2d(ch, inner, 1)
        self.proj = nn.Conv2d(inner, ch, 1)
        
        self.rel_pos = RelativePositionBias(heads, max_h=max_res, max_w=max_res)

    def forward(self, x):
        b, c, h, w = x.shape
        x_n = self.norm(x)
        q = self.to_q(x_n).reshape(b, self.heads, -1, h*w).transpose(-2, -1)
        k = self.to_k(x_n).reshape(b, self.heads, -1, h*w).transpose(-2, -1)
        v = self.to_v(x_n).reshape(b, self.heads, -1, h*w).transpose(-2, -1)
        
        bias = self.rel_pos(h, w).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        
        out = out.transpose(-2, -1).reshape(b, -1, h, w)
        return x + self.proj(out)
    
class MHA_CrossAttention_RelBias(nn.Module):
    def __init__(self, ch, context_ch, heads=4, head_dim=32, max_res=64):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.norm_x = LayerNorm2d(ch)
        self.norm_ctx = LayerNorm2d(context_ch)
        self.to_q = nn.Conv2d(ch, inner_dim, 1)
        self.to_k = nn.Conv2d(context_ch, inner_dim, 1)
        self.to_v = nn.Conv2d(context_ch, inner_dim, 1)
        self.proj = nn.Conv2d(inner_dim, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rel_pos = RelativePositionBias(heads, max_h=max_res, max_w=max_res)
    def forward(self, x, context):
        b, c, h, w = x.shape
        _, _, hc, wc = context.shape
        x_n = self.norm_x(x)
        ctx_n = self.norm_ctx(context)
        q = self.to_q(x_n).view(b, self.heads, self.head_dim, h * w).transpose(-2, -1)
        k = self.to_k(ctx_n).view(b, self.heads, self.head_dim, hc * wc).transpose(-2, -1)
        v = self.to_v(ctx_n).view(b, self.heads, self.head_dim, hc * wc).transpose(-2, -1)
        bias = self.rel_pos(h, w).unsqueeze(0) 
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(-2, -1).reshape(b, -1, h, w)
        return x + self.proj(out)
    
def sinusoidal_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)
    
class UNet_FM(nn.Module):
    def __init__(self, filters_arr, encoder_filters_arr, encoder_denses_arr,
                 t_emb_size, label_emb_size, side_pixels,
                 in_channels=1, in_channels_cond=3, n_channels_group=8,
                 attn=False, cross_attn=False,
                 use_residuals=False, cond_type="concat", encoder_type="simple"):
        super().__init__()

        self.use_residuals = use_residuals
        self.cond_type     = cond_type
        self.use_cross     = cross_attn
        self.encoder_type  = encoder_type
        self.side_pixels   = side_pixels

        decoder_channels = [filters_arr[i] for i in range(len(filters_arr) - 2, -1, -1)]

        if encoder_type == "vit":
            target_sp_ch = decoder_channels[0] if cross_attn else filters_arr[-1]
            self.label_emb = ViT_Encoder(target_sp_ch, label_emb_size, side_pixels)
        elif encoder_type == "resnet":
            enc_dec_ch = decoder_channels if cross_attn else [filters_arr[-1]]
            self.label_emb = ResNet_Encoder(label_emb_size, enc_dec_ch, in_channels_cond, encoder_denses_arr)
        else:
            pass

        emb_dim   = t_emb_size + label_emb_size
        input_dim = (in_channels + in_channels_cond) if cond_type == "concat" else in_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_size, t_emb_size), nn.GELU(),
            nn.Linear(t_emb_size, t_emb_size),
        )

        self.downs = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch  = input_dim if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            layers = nn.ModuleDict({'conv': nn.Conv2d(in_ch, out_ch, 3, padding=1)})
            if use_residuals:
                layers['residual'] = ResidualBlock(out_ch, emb_dim)
            if i < len(filters_arr) - 1:
                layers['downsample'] = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
            self.downs.append(layers)

        self.mid_res1 = ResidualBlock(filters_arr[-1], emb_dim) if use_residuals else nn.Identity()

        num_downsamples = len(filters_arr) - 1
        bottleneck_res = math.ceil(self.side_pixels / (2 ** num_downsamples))
        
        self.mid_attn = MHAAttention(filters_arr[-1], max_res=bottleneck_res) if attn else nn.Identity()

        self.mid_res2 = ResidualBlock(filters_arr[-1], emb_dim) if use_residuals else nn.Identity()

        self.ups = nn.ModuleList()
        for i, dec_ch in enumerate(decoder_channels):
            coarser_idx = len(filters_arr) - 1 - i
            in_ch  = filters_arr[coarser_idx]
            out_ch = dec_ch

            num_downsamples = len(decoder_channels) - 1 - i 
            current_res = math.ceil(self.side_pixels / (2**num_downsamples))

            layers = nn.ModuleDict({
                'upsample_conv': nn.Conv2d(in_ch, out_ch, 3, padding=1),
                'convskip': nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
            })

            if use_residuals: layers['residual'] = ResidualBlock(out_ch, emb_dim)

            is_memory_safe_stage = (i < len(decoder_channels) - 1)
            
            if cross_attn and is_memory_safe_stage:
                ctx_ch = decoder_channels[0] if encoder_type == "vit" else dec_ch
                layers['cross_attn'] = MHA_CrossAttention_RelBias(
                    out_ch, 
                    context_ch=ctx_ch, 
                    max_res=current_res
                )

            self.ups.append(layers)

        self.act_gelu = nn.GELU()

        if self.use_residuals:
            self.refine = ResidualBlock(filters_arr[0], emb_dim)
        else:
            self.refine = nn.Sequential(
                LayerNorm2d(filters_arr[0]),
                nn.Conv2d(filters_arr[0], filters_arr[0], 3, padding=1),
                nn.GELU(),
                nn.Conv2d(filters_arr[0], filters_arr[0], 3, padding=1)
        )

        self.last = nn.Conv2d(filters_arr[0], in_channels, 3, padding=1)
        
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    @torch.amp.autocast('cuda')
    def forward(self, x, t, y, drop_mask=None):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))

        enc_out = self.label_emb(y, drop_mask=drop_mask)
        
        if self.encoder_type == "vit":
            sp_feat, y_global = enc_out
            multi_scale_feats = None
        else:
            multi_scale_feats, y_global = enc_out
            sp_feat = None

        if self.cond_type == "concat": x = torch.cat([x, y], dim=1)

        combined_emb = torch.cat([t_emb, y_global], dim=1)

        skips = []
        for i, down in enumerate(self.downs):
            x = down['conv'](x)
            if self.use_residuals: x = down['residual'](x, combined_emb)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = down['downsample'](x)

        x = self.mid_res1(x, combined_emb) if self.use_residuals else x
        x = self.mid_attn(x)
        x = self.mid_res2(x, combined_emb) if self.use_residuals else x

        for i, up in enumerate(self.ups):
            skip = skips.pop()

            enc_feat = None
            if self.use_cross:
                enc_feat = sp_feat if self.encoder_type == "vit" else multi_scale_feats[i]

            def make_up_block(current_up):
                def up_block_exec(x_in, skip_in, emb_in, enc_in):
                    h = F.interpolate(x_in, size=skip_in.shape[2:], mode='bilinear', align_corners=False)
                    h = current_up['upsample_conv'](h)
                    
                    h = current_up['convskip'](torch.cat([h, skip_in], dim=1))
                    
                    if self.use_cross and enc_in is not None and 'cross_attn' in current_up:
                        if enc_in.shape[2:] != h.shape[2:]:
                            enc_in = F.interpolate(enc_in, size=h.shape[2:], mode='bilinear', align_corners=False)
                        h = current_up['cross_attn'](h, enc_in)

                    if self.use_residuals: h = current_up['residual'](h, emb_in)
                    return h
                return up_block_exec

            up_block_fn = make_up_block(up)

            if self.training:
                x = checkpoint(up_block_fn, x, skip, combined_emb, enc_feat, use_reentrant=False)
            else:
                x = up_block_fn(x, skip, combined_emb, enc_feat)

        if self.use_residuals:
            x = self.refine(x, combined_emb)
        else:
            x = x + self.refine(x)
            
        return self.last(x)
