# ==============================================================================
# Orthogonal Compression for Edge Vision Transformers: Combining Recursive Weight-Sharing with Token Merging
#
# Author: Junseo Kim (UTwente)
# ==============================================================================
# This implementation integrates Token Merging (ToMe) into the Pooling-based Vision Transformer (PiT) architecture.
#
# Upstream Attributions & Core Components:
#   - ToMe (Token Merging): Meta AI (CC BY-NC 4.0)
#   - PiT (Pooling-based Vision Transformer): NAVER AI (Apache-2.0)
#   - PyTorch Image Models (timm): Ross Wightman (Apache-2.0)
# ==============================================================================

import torch
from einops import rearrange
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block as transformer_block
from timm.models.registry import register_model

# * Added imports
from tome import merge
from timm.models.layers import DropPath, Mlp


# * Added a custom attention module to track token mass (size) and return K matrix
class PiT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # * extract the Key matrix and average across heads for ToMe BSM
        k_out = k.mean(dim=1) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # * proportional attention: broadcast size to match heads and add log(size)
        size_broadcast = size.view(B, 1, 1, N)
        attn = attn + torch.log(size_broadcast)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k_out

# * Added a custom transformer block to handle ToMe merging logic
class PiT_Transformer_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PiT_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # * add safe_r constraint
    def get_safe_r(self, sequence_length, target_r):
        """
        Computes a structurally safe token reduction rate that complies with the divisibility constraints and bipartite partitioning limits.

        Args:
            sequence_length: The current number of tokens (N) entering the Transformer block.
            target_r: The requested token reduction target.

        Returns:
            int: The bounded, grid-aligned number of tokens to safely merge.
        """
        if target_r == 0:
            return 0
        # bipartite matching can never merge more than half the available tokens
        max_allowable_r = sequence_length // 2
        return min(target_r, max_allowable_r)

    def forward(self, x, size, target_r, num_cls_tokens):
        attn_out, k_matrix = self.attn(self.norm1(x), size)
        x = x + self.drop_path(attn_out)

        # * calculate safe_r using only the spatial tokens
        spatial_seq_len = x.shape[1] - num_cls_tokens
        safe_r = self.get_safe_r(spatial_seq_len, target_r)

        if safe_r > 0:
            k_spatial = k_matrix[:, num_cls_tokens:] # * isolate the CLS token
            merge_func, unmerge_func = merge.bipartite_soft_matching(k_spatial, safe_r)

            x_cls = x[:, :num_cls_tokens]
            x_spatial = x[:, num_cls_tokens:]
            size_cls = size[:, :num_cls_tokens]
            size_spatial = size[:, num_cls_tokens:]

            # * weighted average merge
            x_spatial_weighted = x_spatial * size_spatial
            x_spatial_summed = merge_func(x_spatial_weighted, mode="sum")
            size_spatial = merge_func(size_spatial, mode="sum")
            x_spatial = x_spatial_summed / size_spatial

            # * recombine the CLS token
            x = torch.cat((x_cls, x_spatial), dim=1)
            size = torch.cat((size_cls, size_spatial), dim=1)
        else:
            unmerge_func = lambda tensor, mode=None: tensor

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size, unmerge_func



# * Modified
class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None,
                 initial_r_ratio=0.0, alpha=1.0, constant_r=None): # * add arguments
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        # * set variables
        self.depth = depth
        self.initial_r_ratio = initial_r_ratio
        self.alpha = alpha
        self.constant_r = constant_r

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        # * switched from timm's block to the custom PiT_Transformer_Block
        self.blocks = nn.ModuleList([
            PiT_Transformer_Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])
        
    # * add a function to get a decaying schedule as specified in the instance attributes
    def get_decaying_schedule(self, initial_r, alpha, depth):
        """
        Generates an exponentially decaying token reduction schedule.
        
        Args:
            initial_r: The starting number of tokens to merge in the first loop.
            alpha: The decay rate (between 0 and 1) at every iteration.
            depth: The total number of recursive loops in the stage.
            
        Returns:
            list: An array of integer 'target_r' values for each loop.
        """
        schedule = []
        
        for d in range(depth):
            r_d = int(math.floor(initial_r * (alpha ** d))) # * apply the decaying formula
            schedule.append(r_d)
            
        return schedule

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        
        # * initialize proportional tracking and unmerge stack
        size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        unmerge_stack = []

        # * dynamic schedule generation based on incoming tokens
        if self.constant_r is not None:
            r_schedule = [self.constant_r] * self.depth
        else:
            # calculate initial_r based ONLY on the spatial tokens (ignore CLS tokens)
            spatial_tokens = x.shape[1] - token_length
            initial_r = int(spatial_tokens * self.initial_r_ratio)
            r_schedule = self.get_decaying_schedule(initial_r, self.alpha, self.depth)

        for i, blk in enumerate(self.blocks):
            target_r = r_schedule[i]
            x, size, unmerge_func = blk(x, size, target_r, num_cls_tokens=token_length)
            
            if target_r > 0:
                unmerge_stack.append(unmerge_func)

        # * separate CLS token before unmerging
        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        
        # * restore original 2D grid dimensions for convolutional pooling
        for unmerge_func in reversed(unmerge_stack):
            x = unmerge_func(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0,
                 initial_r_ratio=0.0, alpha=1.0, constant_r=None): # * add the 'initial_r_ratio', 'alpha', 'constant_r' arguments
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes

        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        # * modify stage logic
        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob,
                            initial_r_ratio=initial_r_ratio, 
                            alpha=alpha,                     
                            constant_r=constant_r) # * pass down arguments    
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)
        cls_token = self.head(cls_token[:, 0])
        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(
            torch.randn(1, 2, self.base_dims[0] * self.heads[0]),
            requires_grad=True)
        if self.num_classes > 0:
            self.head_dist = nn.Linear(self.base_dims[-1] * self.heads[-1],
                                       self.num_classes)
        else:
            self.head_dist = nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls = self.head(cls_token[:, 0])
        x_dist = self.head_dist(cls_token[:, 1])
        if self.training:
            return x_cls, x_dist
        else:
            return (x_cls + x_dist) / 2

@register_model
def pit_b(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_b_820.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def pit_s(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_809.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def pit_xs(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_xs_781.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model

@register_model
def pit_ti(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_ti_730.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def pit_b_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_b_distill_840.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def pit_s_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_s_distill_819.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def pit_xs_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_xs_distill_791.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def pit_ti_distilled(pretrained, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('weights/pit_ti_distill_746.pth', map_location='cpu')
        model.load_state_dict(state_dict)
    return model
