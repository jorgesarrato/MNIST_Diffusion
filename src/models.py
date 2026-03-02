import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models
import math

class ResNet_Encoder(nn.Module):
    def __init__(self, target_spatial_ch, label_emb_size, num_unet_downs, condition_ch = 3, denses_arr=None, return_spatial=True):
        super().__init__()
        self.return_spatial = return_spatial

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if condition_ch == 1:
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode = 'reflect')
            with torch.no_grad():
                resnet.conv1.weight[:] = resnet.conv1.weight.mean(dim=1, keepdim=True)

        children = list(resnet.children())
        
        # Match ResNet downsampling to UNet downsampling
        # UNet downs = 2 -> 1/4 resolution -> ResNet layer1 (idx 5, 64 ch)
        # UNet downs = 3 -> 1/8 resolution -> ResNet layer2 (idx 6, 128 ch)
        # UNet downs = 4 -> 1/16 resolution -> ResNet layer3 (idx 7, 256 ch)
        # UNet downs >= 5 -> 1/32 resolution -> ResNet layer4 (idx 8, 512 ch)
        
        if num_unet_downs <= 2:
            cut_idx = 5
            resnet_ch = 64
        elif num_unet_downs == 3:
            cut_idx = 6
            resnet_ch = 128
        elif num_unet_downs == 4:
            cut_idx = 7
            resnet_ch = 256
        else:
            cut_idx = 8
            resnet_ch = 512
            
        self.backbone = nn.Sequential(*children[:cut_idx])
        
        self.spatial_proj = nn.Conv2d(resnet_ch, target_spatial_ch, kernel_size=1)
        self.norm = nn.GroupNorm(8, target_spatial_ch)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(target_spatial_ch * 16, denses_arr[-1] if denses_arr else 512),
            nn.GELU(),
            nn.Linear(denses_arr[-1] if denses_arr else 512, label_emb_size)
        )

    def forward(self, x):
        if x.shape[1] == 3:
            x = (x - self.mean) / self.std
        feat = self.backbone(x)
        feat = self.spatial_proj(feat)
        feat = self.norm(feat)
            
        global_feat = self.adaptive_pool(feat)
        global_feat = self.fc(self.flatten(global_feat))
        return feat, global_feat

class ResidualBlock(nn.Module):
    def __init__(self, ch, emb_dim, n_channels_group=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(n_channels_group, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, padding_mode = 'reflect')

        if emb_dim > 0:
            self.t_proj = nn.Linear(emb_dim, ch * 2)
            nn.init.zeros_(self.t_proj.weight)
            nn.init.zeros_(self.t_proj.bias)

        self.gn2 = nn.GroupNorm(n_channels_group, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, padding_mode = 'reflect')
        
        self.act_gelu = nn.GELU()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def _impl(self, x, emb):
        h = self.conv1(self.act_gelu(self.gn1(x)))
        if emb is not None:
            gamma, beta = torch.chunk(
                self.t_proj(self.act_gelu(emb))[:, :, None, None], 2, dim=1)
            h = h * (1 + gamma) + beta
        h = self.conv2(self.act_gelu(self.gn2(h)))
        return h + x

    def forward(self, x, emb=None):
        """if self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._impl, x, emb, use_reentrant=False)"""
        return self._impl(x, emb)
    
class Image_Encoder(nn.Module):
    def __init__(self, filters_arr, denses_arr, label_emb_size, in_channels=3, 
                 n_channels_group=8, side_pixels=128):
        super().__init__()
        self.downs = nn.ModuleList()
        
        for i in range(len(filters_arr)):
            in_ch = in_channels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            
            self.downs.append(nn.ModuleDict({
                'conv': nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, padding_mode = 'reflect'),
                'residual': ResidualBlock(out_ch, 0, n_channels_group=n_channels_group)
            }))
            
        self.act_gelu = nn.GELU()

        self.final_norm = nn.GroupNorm(n_channels_group, filters_arr[-1])

        resulting_side_pixels = side_pixels // (2 ** len(filters_arr))
        self.flatten = nn.Flatten()
        self.linears = nn.ModuleList()
        for i in range(len(denses_arr)):
            in_dim = filters_arr[-1] * (resulting_side_pixels**2) if i == 0 else denses_arr[i-1]
            self.linears.append(nn.Linear(in_dim, denses_arr[i]))
        self.last = nn.Linear(denses_arr[-1], label_emb_size)

    def forward(self, x):
        for down in self.downs:
            x = down['conv'](x)
            x = down['residual'](x)
        
        feat = self.act_gelu(self.final_norm(x))
        
        global_feat = self.flatten(x)
        for linear in self.linears:
            global_feat = self.act_gelu(linear(global_feat))
        return feat, global_feat



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
    def __init__(self, ch, context_ch, n_channels_group=8):
        super().__init__()
        self.gn = nn.GroupNorm(n_channels_group, ch)
        
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(context_ch, ch, kernel_size=1)
        self.v = nn.Conv2d(context_ch, ch, kernel_size=1)
        self.final = nn.Conv2d(ch, ch, kernel_size=1)
        
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x, context):
        b, c, h, w = x.shape
        _, c_ctx, h_ctx, w_ctx = context.shape
        
        x_gn = self.gn(x)

        pos_x = get_2d_sincos_pos_embed(c, h, w, x.device)
        pos_context = get_2d_sincos_pos_embed(c_ctx, h_ctx, w_ctx, context.device)

        q = self.q(x_gn + pos_x).view(b, c, -1)
        k = self.k(context + pos_context).view(b, c, -1)
        
        v = self.v(context).view(b, c, -1)

        attn = torch.bmm(q.transpose(1, 2), k) * (c**-0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.view(b, c, h, w)

        return self.final(out) + x
    
class UNet_FM(nn.Module):
    def __init__(self, filters_arr, encoder_filters_arr, encoder_denses_arr, t_emb_size, label_emb_size, side_pixels,
                 in_channels=1, in_channels_cond=3, n_channels_group=8, attn=False, cross_attn=False,
                 use_residuals=False, cond_type="concat", encoder_type="simple"):
        super().__init__()
        
        self.use_residuals = use_residuals
        self.cond_type = cond_type
        self.use_cross = cross_attn

        if encoder_type == "resnet":
            self.label_emb = ResNet_Encoder(encoder_filters_arr[-1], label_emb_size, denses_arr=encoder_denses_arr, condition_ch=in_channels_cond, return_spatial=cross_attn, num_unet_downs=len(filters_arr))
        elif encoder_type == "simple":
            self.label_emb = Image_Encoder(encoder_filters_arr, encoder_denses_arr, label_emb_size, 
                                            in_channels=in_channels_cond, n_channels_group=n_channels_group, 
                                            side_pixels=side_pixels, return_spatial=cross_attn)
        else:
            raise ValueError(f"Encoder type {encoder_type} not supported.")

        if cross_attn:
            self.global_proj = nn.Sequential(
                nn.Linear(encoder_filters_arr[-1], label_emb_size),
                nn.GELU()
            )
        else:
            self.global_proj = nn.Identity()
        
        emb_dim = t_emb_size + label_emb_size

        input_dim = (in_channels + in_channels_cond) if self.cond_type == "concat" else in_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(t_emb_size, t_emb_size),
            nn.GELU(),
            nn.Linear(t_emb_size, t_emb_size)
        )
        
        self.downs = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch = input_dim if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            layers = nn.ModuleDict({'conv': nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode = 'reflect')})
            
            if use_residuals:
                layers['residual'] = ResidualBlock(out_ch, emb_dim, n_channels_group=n_channels_group) 
            else:
                layers['norm'] = nn.GroupNorm(n_channels_group, out_ch)
                layers['ada_proj'] = nn.Linear(emb_dim, out_ch * 2)

            if i < len(filters_arr) - 1:
                layers['downsample'] = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, padding_mode = 'reflect')

            self.downs.append(layers)

        self.mid_res = ResidualBlock(filters_arr[-1], emb_dim, n_channels_group=n_channels_group) if use_residuals else nn.Identity()
        
        if cross_attn:
            self.mid_attn = ResidualCrossAttentionBlock(filters_arr[-1], encoder_filters_arr[-1], n_channels_group)
        else:
            self.mid_attn = ResidualAttentionBlock(filters_arr[-1], n_channels_group) if attn else nn.Identity()

        self.ups = nn.ModuleList()
        for i in range(len(filters_arr)-1, 0, -1):
            out_ch = filters_arr[i-1]
            in_ch = filters_arr[i]
            
            layers = nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                'convskip': nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1, padding_mode = 'reflect')
            })
            layers['residual'] = ResidualBlock(out_ch, emb_dim, n_channels_group=n_channels_group) if use_residuals else \
                                 nn.ModuleDict({'norm': nn.GroupNorm(n_channels_group, out_ch), 'ada_proj': nn.Linear(emb_dim, out_ch * 2)})
            if cross_attn:
                layers['cross_attn'] = ResidualCrossAttentionBlock(
                    out_ch, 
                    encoder_filters_arr[-1],
                    n_channels_group
                )
            self.ups.append(layers)

        self.act_gelu = nn.GELU()
                
        self.last = nn.Conv2d(filters_arr[0], in_channels, kernel_size=3, padding=1, padding_mode = 'reflect')
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features in [f * 2 for f in filters_arr]:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x, t, y):
        t_sinusoidal = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_sinusoidal)
        
        sp_feat, y_global = self.label_emb(y)

        if self.cond_type == "concat":
            x = torch.cat([x, y], dim=1)

        combined_emb = torch.cat([t_emb, y_global], dim=1)

        skips = []
        for i, down in enumerate(self.downs):
            x = down['conv'](x)
            
            if self.use_residuals:
                x = down['residual'](x, combined_emb)
            else:
                x = self.act_gelu(self.apply_adagn(x, combined_emb, down['norm'], down['ada_proj']))
            
            if i < len(self.downs)-1:
                skips.append(x) 
                x = down['downsample'](x)

        x = self.mid_res(x, combined_emb)
        
        if self.use_cross:
            x = self.mid_attn(x, sp_feat)
        else:
            x = self.mid_attn(x)

        for up in self.ups:
            x = up['upsample'](x)
            skip = skips.pop()
            
            if x.shape != skip.shape: 
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear')
                
            x = up['convskip'](torch.cat([x, skip], dim=1))

            if self.use_cross:
                x = up['cross_attn'](x, sp_feat)

            if self.use_residuals:
                x = up['residual'](x, combined_emb)
            else:
                x = self.act_gelu(self.apply_adagn(x, combined_emb, up['norm'], up['ada_proj']))

        return self.last(x)

    def apply_adagn(self, x, emb, norm_layer, proj_layer):
        x = norm_layer(x)
        gamma, beta = torch.chunk(proj_layer(emb)[:, :, None, None], 2, dim=1)
        return x * (1 + gamma) + beta