import torch
import torch.nn as nn

class MotionNet(nn.Module):
    def __init__(self, num_kp=10, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=num_kp * 4, 
            hidden_size=hidden_size, 
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * num_kp)
        )

    def forward(self, kp_source, kp_driving, hidden=None):
        # We assume input is sequenced: (Batch, SeqLen, Features)
        # If we only pass one frame at a time from animate.py, SeqLen=1
        
        # Determine if input has sequence dimension
        has_seq_dim = kp_source.dim() == 3
        
        x = torch.cat([kp_source, kp_driving], dim=-1)
        
        # Add sequence dimension if it's missing (e.g., during frame-by-frame inference)
        if not has_seq_dim:
            x = x.unsqueeze(1)
            
        out, hidden = self.rnn(x, hidden)
        
        # Remove sequence dimension for the Linear layer if we added it
        if not has_seq_dim:
            out = out.squeeze(1)
            
        return self.fc(out), hidden