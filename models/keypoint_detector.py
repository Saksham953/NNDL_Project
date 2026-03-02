import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class KeypointDetector(nn.Module):
    def __init__(self, num_kp=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32, 32),
            nn.MaxPool2d(2), # 64x64
            ResBlock(32, 64),
            nn.MaxPool2d(2), # 32x32
            ResBlock(64, num_kp),
            nn.Conv2d(num_kp, num_kp, 1)
        )

    def forward(self, x):
        heatmaps = self.conv(x)  # (B, K, H, W)
        shape = heatmaps.shape
        heatmaps = heatmaps.view(shape[0], shape[1], -1)
        heatmaps = F.softmax(heatmaps, dim=-1)
        heatmaps = heatmaps.view(shape[0], shape[1], shape[2], shape[3])
        
        # Calculate expected coordinates (x, y)
        grid_x = torch.linspace(-1, 1, shape[3]).view(1, 1, 1, shape[3]).to(x.device)
        grid_y = torch.linspace(-1, 1, shape[2]).view(1, 1, shape[2], 1).to(x.device)
        
        kp_x = (heatmaps * grid_x).sum(dim=(2, 3)) # (B, K)
        kp_y = (heatmaps * grid_y).sum(dim=(2, 3)) # (B, K)
        
        keypoints = torch.stack([kp_x, kp_y], dim=-1) # (B, K, 2)
        
        return keypoints