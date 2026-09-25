import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2D(nn.Module):
    """2D Sinusoidal Positional Encoding for H x W feature maps."""
    def __init__(self, channels: int, height: int = 10, width: int = 10):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f"Channels must be divisible by 4, got {channels}")
        
        pe = torch.zeros(channels, height, width)
        c_half = channels // 2  # e.g. 90
        num_freqs = c_half // 2  # e.g. 45
        
        div_term = torch.exp(torch.arange(0., c_half, 2) * -(math.log(10000.0) / c_half))  # (num_freqs,)
        
        pos_h = torch.arange(0., height).unsqueeze(1)  # (H, 1)
        pos_w = torch.arange(0., width).unsqueeze(1)   # (W, 1)
        
        # Height encodings: shape (H, num_freqs) -> transpose to (num_freqs, H, 1) -> expand to (num_freqs, H, W)
        sin_h = torch.sin(pos_h * div_term).t().unsqueeze(-1).expand(num_freqs, height, width)
        cos_h = torch.cos(pos_h * div_term).t().unsqueeze(-1).expand(num_freqs, height, width)
        
        # Width encodings: shape (W, num_freqs) -> transpose to (num_freqs, 1, W) -> expand to (num_freqs, H, W)
        sin_w = torch.sin(pos_w * div_term).t().unsqueeze(1).expand(num_freqs, height, width)
        cos_w = torch.cos(pos_w * div_term).t().unsqueeze(1).expand(num_freqs, height, width)
        
        pe[0:c_half:2, :, :] = sin_h
        pe[1:c_half:2, :, :] = cos_h
        pe[c_half::2, :, :] = sin_w
        pe[c_half+1::2, :, :] = cos_w
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class AttentionAugmentation2D(nn.Module):
    """2D Multi-Head Spatial Self-Attention Block."""
    def __init__(self, in_channels: int, depth_k: int = 60, depth_v: int = 60, num_heads: int = 2):
        super().__init__()
        self.depth_k = depth_k
        self.depth_v = depth_v
        self.num_heads = num_heads
        self.dk_per_head = depth_k // num_heads
        self.dv_per_head = depth_v // num_heads
        
        self.qkv_conv = nn.Conv2d(in_channels, 2 * depth_k + depth_v, kernel_size=1, bias=True)
        self.scale = self.dk_per_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        qkv = self.qkv_conv(x)
        q, k, v = torch.split(qkv, [self.depth_k, self.depth_k, self.depth_v], dim=1)
        
        # Reshape to (B, num_heads, dk/dv_per_head, N)
        q = q.view(B, self.num_heads, self.dk_per_head, N).permute(0, 1, 3, 2)  # (B, heads, N, dk_per_head)
        k = k.view(B, self.num_heads, self.dk_per_head, N).permute(0, 1, 3, 2)  # (B, heads, N, dk_per_head)
        v = v.view(B, self.num_heads, self.dv_per_head, N).permute(0, 1, 3, 2)  # (B, heads, N, dv_per_head)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)  # (B, heads, N, dv_per_head)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.depth_v, H, W)  # (B, depth_v, H, W)
        return out


class ChrestienHeuristicNet(nn.Module):
    """
    Neural Heuristic Network matching Chrestien et al. NeurIPS 2023.
    Inputs:
      - Current state one-hot: (B, 5, H, W)
      - Goal state one-hot: (B, 5, H, W)
    """
    def __init__(self, dim: int = 10):
        super().__init__()
        self.dim = dim
        in_c = 10  # 5 channels state + 5 channels goal
        
        # 7 Conv layers (64 filters each) with DenseNet-like concat of raw input
        self.conv1 = nn.Conv2d(in_c, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64 + in_c, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64 + in_c, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64 + in_c, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64 + in_c, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64 + in_c, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(64 + in_c, 64, kernel_size=3, padding=1)
        
        # Attention augmented blocks (180 filters each)
        # Block 1
        self.conv8 = nn.Conv2d(64 + in_c, 180, kernel_size=3, padding=1)
        self.att1 = AttentionAugmentation2D(180, depth_k=60, depth_v=60, num_heads=2)
        self.pos1 = PositionalEncoding2D(180, dim, dim)
        
        # Block 2
        # Input channels: 60 (att) + 180 (pos) + 10 (raw input) = 250
        self.conv9 = nn.Conv2d(250, 180, kernel_size=3, padding=1)
        self.att2 = AttentionAugmentation2D(180, depth_k=60, depth_v=60, num_heads=2)
        self.pos2 = PositionalEncoding2D(180, dim, dim)
        
        # Block 3
        self.conv10 = nn.Conv2d(250, 180, kernel_size=3, padding=1)
        self.att3 = AttentionAugmentation2D(180, depth_k=60, depth_v=60, num_heads=2)
        self.pos3 = PositionalEncoding2D(180, dim, dim)
        
        # Block 4
        self.conv11 = nn.Conv2d(250, 180, kernel_size=3, padding=1)
        self.att4 = AttentionAugmentation2D(180, depth_k=60, depth_v=60, num_heads=2)
        self.pos4 = PositionalEncoding2D(180, dim, dim)
        
        # Dense Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(250, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # state, goal: (B, 5, H, W)
        inp = torch.cat([state, goal], dim=1)  # (B, 10, H, W)
        
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(torch.cat([x, inp], dim=1)))
        x = F.relu(self.conv3(torch.cat([x, inp], dim=1)))
        x = F.relu(self.conv4(torch.cat([x, inp], dim=1)))
        x = F.relu(self.conv5(torch.cat([x, inp], dim=1)))
        x = F.relu(self.conv6(torch.cat([x, inp], dim=1)))
        x = F.relu(self.conv7(torch.cat([x, inp], dim=1)))
        
        # Att 1
        p = F.relu(self.conv8(torch.cat([x, inp], dim=1)))
        att15 = self.att1(p)
        att15 = torch.cat([att15, self.pos1(p), inp], dim=1)  # (B, 250, H, W)
        
        # Att 2
        q = F.relu(self.conv9(att15))
        att16 = self.att2(q)
        att16 = torch.cat([att16, self.pos2(q), inp], dim=1)
        
        # Att 3
        r = F.relu(self.conv10(att16))
        att17 = self.att3(r)
        att17 = torch.cat([att17, self.pos3(r), inp], dim=1)
        
        # Att 4
        s = F.relu(self.conv11(att17))
        att18 = self.att4(s)
        att18 = torch.cat([att18, self.pos4(s), inp], dim=1)  # (B, 250, H, W)
        
        # Head
        f2 = self.gap(att18).flatten(1)  # (B, 250)
        d2 = F.relu(self.fc1(f2))
        out = self.fc2(d2)  # (B, 1)
        return out.squeeze(-1)
