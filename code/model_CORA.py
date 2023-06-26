import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
import math


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)        # 0.1
        self.proj_drop = nn.Dropout(drop_p)       # 0.1

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x):
        B, T, C = x.shape   # batch size, seq length, h_dim * n_heads   # [64, 60, 128]

        N, D = self.n_heads, C // self.n_heads   # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)   # [64, 1, 60, 128]

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)       # [64, 1, 60, 60]
        # causal mask applied to weights 权重的因果掩码
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))    # True的位置置-inf
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)     # 权重*V   [64, 1, 60, 128]

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)   # [64, 60, 128]

        out = self.proj_drop(self.proj_net(attention))
        return out       # [64, 60, 128]


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.ln1 = nn.LayerNorm(h_dim)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual   [64, 60, 128]
        x = self.ln1(x)            # LN
        x = x + self.mlp(x)        # residual
        x = self.ln2(x)            # LN
        return x                   # [64, 60, 128]



class CORA(nn.Module):
    def __init__(self, args, final_channels=1):
        super(CORA, self).__init__()
        self.args = args
        self.mlp = nn.Sequential(nn.Linear(512, 4*512),
                                    nn.GELU(),
                                    nn.Linear(4*512, 512),
                                    nn.Dropout(0.2),
            )
        blocks = [Block(512, 16, 4, 0.2) for _ in range(4)]
        self.transformer = nn.Sequential(*blocks)

        self.mlp2 = nn.Sequential(nn.Linear(512, 4*512),
                                    nn.GELU(),
                                    nn.Linear(4*512, 512),
                                    nn.Dropout(0.2),
            )

        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.ln1 = nn.LayerNorm(256)
        self.G_relu = nn.GELU()

        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear_CORA = nn.Linear(256, 3)       # 预留CORR

    def forward(self, x):    # 32, 6, 1024
        x = F.adaptive_max_pool1d(x, 1)     # [64, 512, 1]
        x = x.view(-1, 16, 512)             #  [4, 16, 512]

        x = self.mlp(x) + x
        x = self.transformer(x) + x             # [4, 16, 256]
        # x = self.mlp2(x) + x

        # x = F.leaky_relu(self.ln1(self.linear1(x)), negative_slope=0.2)
        x = self.ln1(self.linear1(x))
        x = self.G_relu(x)
        
        x = self.dp1(x)
        cora = self.linear_CORA(x)         # 4, 16, 4
        cora = F.softmax(cora, dim=-1)     # 归一化
        return cora      # 32, 1

