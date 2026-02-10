import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_FM(nn.Module):
    def __init__(self, filters_arr, t_emb_size, in_channels=1):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_emb_size),
            nn.GELU(),
            nn.Linear(t_emb_size, t_emb_size)
        )

        self.downs = nn.ModuleList()
        self.time_emb_passes = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch = in_channels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            self.downs.append(nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1))
            self.time_emb_passes.append(nn.Linear(t_emb_size, out_ch))

        self.ups = nn.ModuleList()
        self.time_emb_passes_up = nn.ModuleList()
        for i in range(len(filters_arr)-1, 0, -1):
            in_ch = filters_arr[i]
            out_ch = filters_arr[i-1]

            self.ups.append(
                    nn.ModuleDict(
                        {'upsample': nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 2, stride = 2, padding = 0),
                         'convskip': nn.Conv2d(out_ch*2, out_ch, kernel_size = 3, stride = 1, padding = 1)
                        }
                        )
                    )
            self.time_emb_passes_up.append(nn.Linear(t_emb_size, out_ch))

        self.act_gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.last = nn.Conv2d(filters_arr[0], in_channels, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.view(-1, 1))

        skips = []

        for i,down in enumerate(self.downs):
            x = down(x)
            x = self.act_gelu(x + self.time_emb_passes[i](t_emb)[:, :, None, None])
            if i < len(self.downs)-1:
                skips.append(x)
                x = self.pool(x)

        for i,up in enumerate(self.ups):
            x = up['upsample'](x)

            skip = skips.pop()

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

            x = self.act_gelu(up['convskip'](x) + self.time_emb_passes_up[i](t_emb)[:, :, None, None])

        return self.last(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch, t_emb_dim, n_channels_group=8):
        super().__init__()
        self.gn1 = nn.GroupNorm(n_channels_group, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1)

        self.t_proj = nn.Linear(t_emb_dim, ch)

        self.gn2 = nn.GroupNorm(n_channels_group, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size = 3, stride = 1, padding = 1)
        
        self.act_gelu = nn.GELU()

    def forward(self, x, t_emb):
        x_in = x
        x = self.conv1(self.act_gelu(self.gn1(x_in)))
        
        x = x + self.t_proj(t_emb)[:, :, None, None] 
        
        x = self.conv2(self.act_gelu(self.gn2(x)))
        return x + x_in

class UNet_FM_Residuals(nn.Module):
    def __init__(self, filters_arr, t_emb_size, in_channels=1, n_channels_group=8):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_emb_size),
            nn.GELU(),
            nn.Linear(t_emb_size, t_emb_size)
        )

        self.downs = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch = in_channels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            self.downs.append(
                nn.ModuleDict(
                    {'conv': nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1),
                    'residual':ResidualBlock(out_ch, t_emb_size, n_channels_group=n_channels_group)
                    }
                    )
                )
                

        self.ups = nn.ModuleList()
        for i in range(len(filters_arr)-1, 0, -1):
            in_ch = filters_arr[i]
            out_ch = filters_arr[i-1]

            self.ups.append(
                    nn.ModuleDict(
                        {'upsample': nn.ConvTranspose2d(in_ch, out_ch, kernel_size = 2, stride = 2, padding = 0),
                         'convskip': nn.Conv2d(out_ch*2, out_ch, kernel_size = 3, stride = 1, padding = 1),
                         'residual': ResidualBlock(out_ch, t_emb_size)
                        }
                        )
                    )

        self.act_gelu = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.last = nn.Conv2d(filters_arr[0], in_channels, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.view(-1, 1))

        skips = []

        for i,down in enumerate(self.downs):
            x = down['conv'](x)
            x = down['residual'](x, t_emb)
            if i < len(self.downs)-1:
                skips.append(x)
                x = self.pool(x)

        for i,up in enumerate(self.ups):
            x = up['upsample'](x)

            skip = skips.pop()

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

            x = up['convskip'](x)
            x = up['residual'](x, t_emb)

        return self.last(x)
