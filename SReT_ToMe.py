# ==============================================================================
# Orthogonal Compression for Edge Vision Transformers: Combining Recursive Weight-Sharing with Token Merging
#
# Author: Junseo Kim (UTwente)
# ==============================================================================
# This implementation integrates Token Merging (ToMe) into the Sliced Recursive Transformer (SReT) architecture
# to optimize parameter storage, peak activation memory, and throughput.
#
# Upstream Attributions & Core Components:
#   - ToMe (Token Merging): Meta AI (CC BY-NC 4.0)
#   - SReT (Sliced Recursive Transformer): Zhiqiang Shen (MIT)
#   - PiT (Spatial Dimensions of Vision Transformers): NAVER Corp (Apache-2.0)
#   - PyTorch Image Models (timm): Ross Wightman (Apache-2.0)
# ==============================================================================

import torch
from einops import rearrange
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple, lecun_normal_
from timm.models.registry import register_model

# * Added import
from tome import merge

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out

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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Non_proj(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()

    def forward(self, x, recursive_index):
        x = self.coefficient1(x) + self.coefficient2(self.mlp(self.norm1(x)))
        return x


class Group_Attention(nn.Module):
    def __init__(self, dim, num_groups1=8, num_groups2=4, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups1 = num_groups1
        self.num_groups2 = num_groups2
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # * Modified
    def forward(self, x, size, recursive_index): # * added 'size' argument to track merged token mass for porportional attention
        B, N, C = x.shape
        if recursive_index == False:
            num_groups = self.num_groups1
        else:
            num_groups = self.num_groups2
            if num_groups != 1:
                idx = torch.randperm(N)
                x = x[:,idx,:]
                size = size[:,idx,:] # * also scramble the 'size' tensor
                inverse = torch.argsort(idx)
        qkv = self.qkv(x).reshape(B, num_groups, N // num_groups, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)  
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # * get the 'Key' matrix by averaging over attention heads to run the ToMe bipartite matching algorithm
        k_out = k.mean(dim=2).reshape(B, N, -1)

        # * reshape the 'size' tensor to match the attention heads (shape: [Batch, Groups, 1 (broadcast heads), 1 (broadcast queries), Tokens Per Group])
        size_grouped = size.reshape(B, num_groups, 1, 1, N // num_groups)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn + torch.log(size_grouped) # * add 'log(size)' before softmax just like ToMe

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(2, 3).reshape(B, num_groups, N // num_groups, C)
        x = x.permute(0, 3, 1, 2).reshape(B, C, N).transpose(1, 2)
        if recursive_index == True and num_groups != 1:
            x = x[:,inverse,:]
            k_out = k_out[:,inverse,:] # * unscramble the 'Key' matrix also
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, k_out # * also return 'Key' matrix

# * Modified
class Transformer_Block(nn.Module):

    def __init__(self, dim, num_groups1, num_groups2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Group_Attention(
            dim, num_groups1=num_groups1, num_groups2=num_groups2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.coefficient1 = LearnableCoefficient()
        self.coefficient2 = LearnableCoefficient()
        self.coefficient3 = LearnableCoefficient()
        self.coefficient4 = LearnableCoefficient()

    # * add a function to always get a safe value of 'r' (merge rate)
    def get_r(self, sequence_length, target_r, group_size):
        """
        Computes a structurally safe token reduction rate that complies with the divisibility constraints and bipartite partitioning limits.

        Args:
            sequence_length: The current number of tokens (N) entering the Transformer block.
            target_r: The requested token reduction target.
            group_size: The least common multiple (LCM) of the group configurations for the current stage.

        Returns:
            int: The bounded, grid-aligned number of tokens to safely merge.
        """
        if target_r == 0:
            return 0
            
        # * the initial target sequence length
        initial_length = sequence_length - target_r 
        
        # * ensure that the sequence length is a clean multiple of 'group_size'
        safe_length = (initial_length // group_size) * group_size 
        
        # * ensure that the sequence length does not fall below the 'group_size' (needs at least 1 token per sliced attention)
        safe_length = max(safe_length, group_size) 
        
        # * compute the 'safe_r' actually needed to achieve a safe sequence length
        safe_r = sequence_length - safe_length 

        # * safety to ensure that 'r' is not negative
        safe_r = max(0, safe_r)
    
        # * since bipartite soft matching of ToMe divides the tokens into 2 groups before matching, 'r' cannot be more than the number of tokens in any group
        max_allowable_r = sequence_length // 2
        safe_r = min(safe_r, max_allowable_r)
        
        return safe_r

    # * Modified
    def forward(self, x, size, recursive_index, group_size, target_r, source=None): # * added 'size', 'group_size', 'target_r', 'source' arguments
        b, n, c = x.shape # * get batch size, sequence length, and dimensions of x (b,n,c)
        safe_r = self.get_r(n, target_r, group_size) # * get 'safe_r' using given arguments
        
        attn_out, k_matrix = self.attn(self.norm1(x), size, recursive_index) # * apply proportional attention
        x = self.coefficient1(x) + self.coefficient2(self.drop_path(attn_out))

        if safe_r > 0:
            merge_func, unmerge_func = merge.bipartite_soft_matching(k_matrix, safe_r) # * use the 'Key' matrix to get the merge and unmerge functions
        else:
            def placeholder(tensor, mode=None):
                return tensor
            merge_func = placeholder
            unmerge_func = placeholder

        # * update the visual trace matrix
        if source is not None:
            source = merge_func(source, mode="amax")

        # * pre weight the features by their tracked mass
        x_weighted = x * size
        
        # * sum the features to compute the weighted average
        x_summed = merge_func(x_weighted, mode="sum")

        # * sum the 'size' tensor to track merged token mass for downstream blocks
        size = merge_func(size, mode="sum")
        
        # * divide by the tracked mass to compute the weighted average
        x = x_summed / size

        x = self.coefficient3(x) + self.coefficient4(self.drop_path(self.mlp(self.norm2(x))))
        return x, size, unmerge_func, source # * also return the 'size' tensor, 'unmerge' function, and 'source' matrix

    
# * Modified
class Transformer(nn.Module):
    def __init__(self, base_dim, depth, recursive_num, groups1, groups2, heads, mlp_ratio, np_mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None, initial_r_ratio=0.0, alpha=1.0, constant_r=None): # * add the 'initial_r_ratio', 'alpha', 'constant_r' arguments to also for a variable decaying schedule
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        blocks = [
            Transformer_Block(
                dim=embed_dim,
                num_groups1=groups1,
                num_groups2=groups2,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(recursive_num)]

        recursive_loops = int(depth/recursive_num)
        non_projs = [
            Non_proj(
                dim=embed_dim, num_heads=heads, mlp_ratio=np_mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=drop_path_prob[i], norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
            for i in range(depth)]
        RT = []
        for rn in range(recursive_num):
            for rl in range(recursive_loops):
                RT.append(blocks[rn])
                RT.append(non_projs[rn*recursive_loops+rl])

        self.blocks = nn.ModuleList(RT)

        # * intiialize required instance attributes
        self.group1 = groups1
        self.group2 = groups2
        self.depth = depth
        self.initial_r_ratio = initial_r_ratio
        self.alpha = alpha
        self.constant_r = constant_r
    
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

    # * Modified
    def forward(self, x, trace_source=False): # * add 'trace_source' argument
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        unmerge_stack = [] # * initialize the unmerge stack used to match tensor shapes for the convlutional pooling layer in between stages

        # * initialize the 'size' tensor to keep track of the mass of the merged tokens for propotional attention
        size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        # * compute the LCM of the number of groups within the stage to achieve safe token reduction within the block
        stage_lcm = math.lcm(self.group1, self.group2)

        # * create the N x N tracking matrix
        if trace_source:
            source = torch.eye(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            source = None

        # * apply the reduction schedule
        if self.constant_r is not None:
            # * constant reduction
            r_schedule = [self.constant_r] * self.depth
        else:
            # * dynamic reduction
            initial_r = int(x.shape[1] * self.initial_r_ratio) 
            r_schedule = self.get_decaying_schedule(initial_r=initial_r, alpha=self.alpha, depth=self.depth)

        transfomer_idx = 0
        for i, blk in enumerate(self.blocks):

            if isinstance(blk, Transformer_Block):
                # * only run for the transformer blocks

                if (i+2)%4 == 0:
                    recursive_index = True
                else:
                    recursive_index = False
                
                target_r = r_schedule[transfomer_idx] # * get the corresponding 'target_r' for the block index

                # * pass and receive the 'source' variable
                x, size, unmerge_func, source = blk(x, size, recursive_index, stage_lcm, target_r, source=source) # * run the transformer block

                unmerge_stack.append(unmerge_func) # * store the unmerge functions

                transfomer_idx += 1
            
            else:
                # * pass through normally for non-transformer blocks
                x = blk(x, recursive_index=False)

        # * save final state before restoring grid
        if trace_source:
            self._tome_info = {"source": source}

        # * restore original tensor size just for the convolution pooling layer by applying the stored 'unmerge' functions in reverse
        for unmerge_func in reversed(unmerge_stack):
            x = unmerge_func(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)

    def forward(self, x):

        x = self.conv(x)

        return x


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, int(out_channels/2), kernel_size=3,
                              stride=2, padding=1, bias=True)
        self.bn1 = norm_layer(int(out_channels/2))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(int(out_channels/2), out_channels, kernel_size=3,
                              stride=2, padding=1, bias=True)
        self.bn2 = norm_layer(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=2, padding=1, bias=True)
        self.bn3 = norm_layer(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x


# * Modified
class SReT(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, recursive_num, groups1, groups2, heads,
                 mlp_ratio, np_mlp_ratio, num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.1, initial_r_ratio=0.0, alpha=1.0, constant_r=None): # * add the 'initial_r_ratio', 'alpha', 'constant_r' arguments to also for a variable decaying schedule
        super(SReT, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = int(image_size/8)

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

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], recursive_num[stage], groups1[stage], groups2[stage], heads[stage],
                            mlp_ratio, np_mlp_ratio, 
                            drop_rate, attn_drop_rate, drop_path_prob, initial_r_ratio=initial_r_ratio, alpha=alpha, constant_r=constant_r)
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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
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

        for stage in range(len(self.pools)):
            x = self.transformers[stage](x)
            x = self.pools[stage](x)
        x = self.transformers[-1](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Distilled_SReT(SReT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.forward_features(x)
        x_cls = self.head(x)
        # `x_cls, x_cls` is used to make it compatible with DeiT codebase, while SReT uses global_average pooling, and soft label only for knowledge distillation
        # so `x_cls` is enough
        if self.training:
            # return x_cls, x_cls
            return x_cls
        else:
            return x_cls


@register_model
def SReT_T(pretrained=False, **kwargs):
    model = SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2,5,3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=3.6,
        np_mlp_ratio=1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_T.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

@register_model
def SReT_LT(pretrained=False, **kwargs):
    model = SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1], # [16, 14, 1] 
        groups2=[2, 1, 1], # [1, 1, 1]
        mlp_ratio=4.0,
        np_mlp_ratio=1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_LT.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

def SReT_S(pretrained=False, **kwargs):
    model = SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[42, 42, 42],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[3, 6, 12],
        groups1=[8, 4, 1], 
        groups2=[2, 1, 1],
        mlp_ratio=3.0,
        np_mlp_ratio=2,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_S.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

# Knowledge Distillation
@register_model
def SReT_T_distill(pretrained=False, **kwargs):
    model = Distilled_SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=3.6,
        np_mlp_ratio=1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_T_distill.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

@register_model
def SReT_LT_distill(pretrained=False, **kwargs):
    model = Distilled_SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[2, 4, 8],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=4.0,
        np_mlp_ratio=1,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_LT_distill.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model

def SReT_S_distill(pretrained=False, **kwargs):
    model = Distilled_SReT(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[42, 42, 42],
        depth=[4, 10, 6],
        recursive_num=[2, 5, 3],
        heads=[3, 6, 12],
        groups1=[8, 4, 1],
        groups2=[2, 1, 1],
        mlp_ratio=3.0,
        np_mlp_ratio=2,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('SReT_S_distill.pth', map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model