import torch
import torch.nn as nn

class Ublock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Ublock, self).__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    out = self.block1(x)
    return out

class Unet(nn.Module):
  def __init__(self):
    super(Unet, self).__init__()
    self.encoder1 = Ublock(3, 64)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder2 = Ublock(64, 128)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder3 = Ublock(128, 256)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder4 = Ublock(256, 512)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.bottleneck = Ublock(512, 1024)
    self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.decoder4 = Ublock(1024, 512)
    self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.decoder3 = Ublock(512, 256)
    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.decoder2 = Ublock(256, 128)
    self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.decoder1 = Ublock(128, 64)
    self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

  def forward(self, x):
    enc1 = self.encoder1(x)
    pool1 = self.pool1(enc1)

    enc2 = self.encoder2(pool1)
    pool2 = self.pool2(enc2)

    enc3 = self.encoder3(pool2)
    pool3 = self.pool3(enc3)

    enc4 = self.encoder4(pool3)
    pool4 = self.pool4(enc4)

    bottleneck = self.bottleneck(pool4)

    upconv4 = self.upconv4(bottleneck)
    concat4 = torch.cat([upconv4, enc4], dim=1)
    dec4 = self.decoder4(concat4)

    upconv3 = self.upconv3(dec4)
    concat3 = torch.cat([upconv3, enc3], dim=1)
    dec3 = self.decoder3(concat3)

    upconv2 = self.upconv2(dec3)
    concat2 = torch.cat([upconv2, enc2], dim=1)
    dec2 = self.decoder2(concat2)

    upconv1 = self.upconv1(dec2)
    concat1 = torch.cat([upconv1, enc1], dim=1)
    dec1 = self.decoder1(concat1)

    output = self.final_conv(dec1)
    return output
