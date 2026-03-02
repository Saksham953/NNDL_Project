import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # input is (3) x 128 x 128 -> (64) x 64 x 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        
        # (64) x 64 x 64 -> (128) x 32 x 32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        
        # (128) x 32 x 32 -> (256) x 16 x 16
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        
        # (256) x 16 x 16 -> (512) x 16 x 16 (stride 1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        
        # (512) x 16 x 16 -> 1 x 15 x 15 PatchGAN output
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        out = self.lrelu1(self.conv1(x))
        out = self.lrelu2(self.bn2(self.conv2(out)))
        out = self.lrelu3(self.bn3(self.conv3(out)))
        out = self.lrelu4(self.bn4(self.conv4(out)))
        out = self.conv5(out)
        return out
