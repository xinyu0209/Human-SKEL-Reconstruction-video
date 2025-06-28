import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .vit import PatchEmbed
from mamba_ssm.modules.mamba_simple import Mamba

class SpatialConvEnhancer(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size, padding=kernel_size//2, groups=embed_dim)
        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x, Hp, Wp):
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, Hp, Wp)
        x = self.act(self.bn(self.conv(x)))
        x = x.flatten(2).transpose(1, 2)
        return x

class Mamba2(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 drop_rate=0., enhancer_kernel=3, norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad',
                 **kwargs):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.depth = depth

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.enhancer = SpatialConvEnhancer(embed_dim, kernel_size=enhancer_kernel)
        self.blocks = nn.ModuleList([
            Mamba(d_model=embed_dim, **kwargs)
            for _ in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.enhancer(x, Hp, Wp)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
