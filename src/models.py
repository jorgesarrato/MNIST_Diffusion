import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_FM(nn.Module):
    def __init__(self, in_channels, filters_arr, t_emb_size):
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

            curr_channels = out_ch

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

        self.last = nn.Conv2d(filters_arr[i-1], in_channels, kernel_size = 3, stride = 1, padding = 1)

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
