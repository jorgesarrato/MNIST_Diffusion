import torch
import torch.nn as nn
import torch.optim as optim

class UNet_FM(nn.Module):
    def __init__(self, in_channels, filters_arr, kerne_size, stride, padding):
        super().__init__()

        self.downs = nn.ModuleList()
        for i in range(len(filters_arr)):
            in_ch = in_chanels if i == 0 else filters_arr[i-1]
            out_ch = filters_arr[i]
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
                    nn.GELU(),
                )
            )

            curr_channels = filters

        self.ups = nn.ModuleList()
        for i in range(len(filters_arr), -1):
            in_ch = filterrs_arr[i]
            out_ch = filters_arr[i-1]

            #TODO: upsampling and convolve with skips

            #TODO: time embeddings

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)

        #TODO: pass through decoder

        return x
