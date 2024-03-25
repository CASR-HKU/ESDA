from collections import OrderedDict

import torch
import torch.nn as nn
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
import MinkowskiEngine as ME


class UNetSubmanifold(nn.Module):

    def __init__(self, args):
        super(UNetSubmanifold, self).__init__()
        in_channels = 3
        out_channels = 1
        init_features = 32
        features = init_features
        self.use_heatmap = True
        self.encoder1 = UNetSubmanifold._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetSubmanifold._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetSubmanifold._block(features * 2, features * 4, name="bottleneck")

        # self.upconv4 = nn.ConvTranspose2d(
        #     features * 16, features * 8, kernel_size=2, stride=2
        # )
        # self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        # self.upconv3 = nn.ConvTranspose2d(
        #     features * 8, features * 4, kernel_size=2, stride=2
        # )
        # self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolutionTranspose

        self.upconv2 = MinkowskiConvolutionTranspose(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNetSubmanifold._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = MinkowskiConvolutionTranspose(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNetSubmanifold._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        # enc3 = self.encoder3(self.pool2(enc2))
        # enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool2(enc2))

        # dec4 = self.upconv4(bottleneck)
        # dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)
        # dec3 = self.upconv3(bottleneck)
        # dec3 = torch.cat((dec3, enc3), dim=1)
        # dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        output = self.conv(dec1)
        # return torch.sigmoid(self.conv(dec1))
        return output.view(batch_size, seq_len, height, width).permute(0, 1, 3, 2)


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        ME.MinkowskiConvolution(
                            in_channels,
                            features,
                            kernel_size=3,
                            stride=1,
                            dimension=2,
                        )
                    ),
                    (name + "norm1", ME.MinkowskiBatchNorm(num_features=features)),
                    (name + "relu1", ME.MinkowskiReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", ME.MinkowskiBatchNorm(num_features=features)),
                    (name + "relu2", ME.MinkowskiReLU(inplace=True)),
                ]
            )
        )
