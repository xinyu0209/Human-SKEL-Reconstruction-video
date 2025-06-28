import torch
import torch.nn as nn
from functools import partial
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.norm = norm_layer(dim)
        
        # 使用简化版的Mamba结构
        self.in_proj = nn.Linear(dim, dim * expand, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=dim * expand,
            out_channels=dim * expand,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=dim * expand,
            bias=False
        )
        self.x_proj = nn.Linear(dim * expand, d_state + d_state, bias=False)
        self.dt_proj = nn.Linear(d_state, dim * expand, bias=True)  # 修正此处
        self.A_proj = nn.Linear(d_state, dim * expand, bias=True)  # 新增
        self.out_proj = nn.Linear(dim * expand, dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        B, N, C = x.shape
        
        # 残差连接
        shortcut = x
        x = self.norm(x)
        
        # 投影扩展
        x = self.in_proj(x)  # (B, N, C*expand)
        
        # 1D卷积
        x = x.transpose(1, 2)  # (B, C*expand, N)
        x = self.conv1d(x)[:, :, :N]  # 因果卷积
        x = x.transpose(1, 2)  # (B, N, C*expand)
        
        # Mamba核心操作
        x = self.act(x)
        x_dbl = self.x_proj(x)  # (B, N, d_state+d_state)
        
        # 这里简化了SSM操作，实际Mamba会更复杂
        assert x_dbl.shape[-1] == 2 * self.d_state, f"split前shape不符: got {x_dbl.shape[-1]}, expect {2*self.d_state}"
        dt, A = torch.split(x_dbl, [self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        A = self.A_proj(A)           # (B, N, dim*expand)
        x = x * torch.sigmoid(dt) + A
        
        x = self.out_proj(x)
        x = shortcut + self.drop_path(x)
        return x

class PatchEmbed(nn.Module):
    """与ViT相同的Patch Embedding层"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                             stride=(patch_size[0] // ratio), 
                             padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Mamba(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.depth = depth

        # Patch embedding (与ViT相同)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # 位置编码 (可以保持与ViT相同)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 使用Mamba块替换Transformer块
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=embed_dim, 
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                drop_path=dpr[i], 
                norm_layer=norm_layer
            )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """冻结参数 (与ViT相同)"""
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
        """初始化权重 (与ViT相同)"""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # 保持与ViT相同的位置编码方式
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        # 保持与ViT相同的输出形状
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """转换为训练模式 (与ViT相同)"""
        super().train(mode)
        self._freeze_stages()