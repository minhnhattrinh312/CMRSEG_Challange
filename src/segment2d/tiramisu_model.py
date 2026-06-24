import torch
import torch.nn as nn


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module("norm", nn.GroupNorm(num_groups=1, num_channels=in_channels))
        self.add_module("silu", nn.SiLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.add_module("drop", nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module("norm", nn.GroupNorm(num_groups=1, num_channels=in_channels))
        self.add_module("SiLU", nn.SiLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.add_module("drop", nn.Dropout2d(0.2))
        self.add_module("maxpool", nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=True,
        )

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module("bottleneck", DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2 : (xy2 + max_height), xy1 : (xy1 + max_width)]




class FinalDecoderHead(nn.Module):
    def __init__(self, in_channels, mid_channels, n_classes, dropout=0.2):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.Dropout2d(dropout),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiHeadFCDenseNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5),
        bottleneck_layers=5,
        growth_rate=12,
        out_chans_first_conv=48,
        head_classes=None,
        final_head_mid_channels=64,
    ):
        super().__init__()

        if head_classes is None:
            head_classes = {
                "SAX": 4,
                "2CH": 3,
                "4CH": 6,
            }

        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.head_classes = head_classes

        cur_channels_count = out_chans_first_conv
        skip_connection_channel_counts = []

        self.firstconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_chans_first_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.denseBlocksDown = nn.ModuleList()
        self.transDownBlocks = nn.ModuleList()

        for n_layers in down_blocks:
            self.denseBlocksDown.append(DenseBlock(cur_channels_count, growth_rate, n_layers))
            cur_channels_count += growth_rate * n_layers

            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        self.bottleneck = Bottleneck(
            cur_channels_count,
            growth_rate,
            bottleneck_layers,
        )

        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        self.transUpBlocks = nn.ModuleList()
        self.denseBlocksUp = nn.ModuleList()

        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))

            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(
                    cur_channels_count,
                    growth_rate,
                    up_blocks[i],
                    upsample=True,
                )
            )

            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))

        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(
            DenseBlock(
                cur_channels_count,
                growth_rate,
                up_blocks[-1],
                upsample=False,
            )
        )

        cur_channels_count += growth_rate * up_blocks[-1]

        self.finalHeads = nn.ModuleDict(
            {
                view: FinalDecoderHead(
                    in_channels=cur_channels_count,
                    mid_channels=final_head_mid_channels,
                    n_classes=n_classes,
                )
                for view, n_classes in head_classes.items()
            }
        )

        self.softmax = nn.Softmax(dim=1)

    def shared_forward(self, x):
        out = self.firstconv(x)

        skip_connections = []

        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        return out

    def forward(self, x, view):
        if view not in self.finalHeads:
            raise ValueError(f"Unknown view '{view}'. Available views: {list(self.finalHeads.keys())}")

        out = self.shared_forward(x)
        out = self.finalHeads[view](out)
        out = self.softmax(out)

        return out
