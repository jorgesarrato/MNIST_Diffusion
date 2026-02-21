import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch, emb_dim, n_channels_group=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(n_channels_group, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1)

        if emb_dim > 0:
            self.t_proj = nn.Linear(emb_dim, ch)

        self.gn2 = nn.GroupNorm(n_channels_group, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1)
        
        self.act_gelu = nn.GELU()

    def forward(self, x, emb=None):
        x_in = x
        x = self.conv1(self.act_gelu(self.gn1(x_in)))

        if emb is not None:
            x = x + self.t_proj(emb)[:, :, None, None] 
        
        x = self.conv2(self.act_gelu(self.gn2(x)))
        return x + x_in
    
class Image_Encoder(nn.Module):
    def __init__(self, filters_arr, denses_arr, label_emb_size, in_channels=3, n_channels_group=8, side_pixels = 128):
        super().__init__()

        self.downs = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch = in_channels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            self.downs.append(
                nn.ModuleDict(
                    {'conv': nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
                    'residual':ResidualBlock(out_ch, 0, n_channels_group=n_channels_group)
                    }))
            
        resulting_side_pixels = side_pixels // (2 ** len(filters_arr))
        

        self.flatten = nn.Flatten()
        
        self.linears = nn.ModuleList()
        for i in range(len(denses_arr)):
            in_dim = filters_arr[-1] * resulting_side_pixels**2 if i == 0 else denses_arr[i-1]
            out_dim = denses_arr[i]
            self.linears.append(nn.Linear(in_dim, out_dim))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act_gelu = nn.GELU()

        self.last = nn.Linear(denses_arr[-1], label_emb_size)

    def forward(self, x):
        for down in self.downs:
            x = down['conv'](x)
            x = down['residual'](x)
            x = self.pool(x)
        
        x = self.flatten(x)
        for linear in self.linears:
            x = self.act_gelu(linear(x))
            
        return self.last(x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, ch, n_channels_group=8):
        super().__init__()
        self.gn = nn.GroupNorm(n_channels_group, ch)
        self.k = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
        self.q = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
        self.v = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)
        self.final = nn.Conv2d(ch, ch, kernel_size = 1, stride = 1, padding = 0)

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

class UNet_FM(nn.Module):
    def __init__(self, filters_arr, encoder_filters_arr, encoder_denses_arr, t_emb_size, label_emb_size, side_pixels,
                 in_channels=1, in_channels_cond=1, n_channels_group=8, attn=False, use_residuals=False):
        super().__init__()
        
        self.use_residuals = use_residuals
        self.label_emb = Image_Encoder(encoder_filters_arr, encoder_denses_arr, label_emb_size, 
                                     in_channels=in_channels_cond, n_channels_group=n_channels_group, side_pixels=side_pixels)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_emb_size),
            nn.GELU(),
            nn.Linear(t_emb_size, t_emb_size)
        )
        
        emb_dim = t_emb_size + label_emb_size

        self.downs = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch = in_channels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            
            layers = nn.ModuleDict({'conv': nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)})
            
            if self.use_residuals:
                layers['residual'] = ResidualBlock(out_ch, emb_dim, n_channels_group=n_channels_group)
            else:
                layers['norm'] = nn.GroupNorm(n_channels_group, out_ch)
                layers['ada_proj'] = nn.Linear(emb_dim, out_ch * 2)
                
            self.downs.append(layers)

        if self.use_residuals:
            self.mid_res = ResidualBlock(filters_arr[-1], emb_dim, n_channels_group=n_channels_group)
        else:
            self.mid_res = nn.Identity()
            
        self.mid_attn = ResidualAttentionBlock(filters_arr[-1], n_channels_group=n_channels_group) if attn else nn.Identity()

        self.ups = nn.ModuleList()
        for i in range(len(filters_arr)-1, 0, -1):
            in_ch = filters_arr[i]
            out_ch = filters_arr[i-1]

            layers = nn.ModuleDict({
                'upsample': nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                'convskip': nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1)
            })
            
            if self.use_residuals:
                layers['residual'] = ResidualBlock(out_ch, emb_dim, n_channels_group=n_channels_group)
            else:
                layers['norm'] = nn.GroupNorm(n_channels_group, out_ch)
                layers['ada_proj'] = nn.Linear(emb_dim, out_ch * 2)
                
            self.ups.append(layers)

        self.act_gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last = nn.Conv2d(filters_arr[0], in_channels, kernel_size=3, padding=1)

    def apply_adagn(self, x, emb, norm_layer, proj_layer):
        x = norm_layer(x)

        ada_params = proj_layer(emb)[:, :, None, None]
        gamma, beta = torch.chunk(ada_params, 2, dim=1)
        return x * (1 + gamma) + beta

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t.view(-1, 1))
        label_emb = self.label_emb(y)
        combined_emb = torch.cat([t_emb, label_emb], dim=1)

        skips = []

        for i, down in enumerate(self.downs):
            x = down['conv'](x)
            
            if self.use_residuals:
                x = down['residual'](x, combined_emb)
            else:
                x = self.apply_adagn(x, combined_emb, down['norm'], down['ada_proj'])
                x = self.act_gelu(x)
                
            if i < len(self.downs)-1:
                skips.append(x)
                x = self.pool(x)

        x = self.mid_res(x, combined_emb) if self.use_residuals else self.mid_res(x)
        x = self.mid_attn(x)

        for i, up in enumerate(self.ups):
            x = up['upsample'](x)
            skip = skips.pop()

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = up['convskip'](x)

            if self.use_residuals:
                x = up['residual'](x, combined_emb)
            else:
                x = self.apply_adagn(x, combined_emb, up['norm'], up['ada_proj'])
                x = self.act_gelu(x)

        return self.last(x)