import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    """ Transformer Encoder Layer with checkpointing support. """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class DropPath(nn.Module):
    """Stochastic depth layer"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1.0 - self.drop_prob
            mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob
            return x * mask / keep_prob
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, num_classes=3, 
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_c, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.2)  # Increased dropout
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0.2, depth)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False)  # ✅ Ensures proper gradient flow

        x = self.norm(x)
        return self.head(x[:, 0])
