import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_net import TransformerNet

class Generator(nn.Module):
    def __init__(self, num_kp=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )

        self.transformer = TransformerNet(feature_dim=2 * num_kp)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128 + 2 * num_kp, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1) # Predicts 2D Flow (dx, dy)
        )
        
        # Initialize output to zero flow so we start with a perfectly sharp, motionless identity mapping
        self.decoder[-1].weight.data.zero_()
        self.decoder[-1].bias.data.zero_()

    def forward(self, x, motion):
        f = self.encoder(x)
        
        # Apply transformer if motion is a sequence
        if motion.dim() == 3:
            motion = self.transformer(motion)
            motion = motion[:, -1, :]
            
        # Tile motion to match the spatial dimensions of the encoded feature map
        b, c, h, w = f.shape
        motion_tiled = motion.view(b, -1, 1, 1).expand(b, -1, h, w)
        
        # Concatenate encoded features with motion features
        f = torch.cat([f, motion_tiled], dim=1)
        
        flow = self.decoder(f) # Flow field of shape (B, 2, H, W)
        
        # Create base coordinate grid
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1, 1, x.shape[2], device=x.device),
            torch.linspace(-1, 1, x.shape[3], device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([base_x, base_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        
        # Add predicted flow to grid (permute flow to shape (B, H, W, 2))
        grid = base_grid + flow.permute(0, 2, 3, 1)
        
        # Warp the original sharp image using the flow field
        out = F.grid_sample(x, grid, align_corners=True, padding_mode="border")
        return out