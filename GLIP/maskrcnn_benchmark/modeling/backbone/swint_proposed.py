# --------------------------------------------------------
# Swin Transformer
# modified from https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/blob/master/mmdet/models/backbones/swin_transformer.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swint import (Mlp, window_partition, window_reverse, WindowAttention, \
    SwinTransformerBlock , PatchMerging , BasicLayer, PatchEmbed, SwinTransformer)

import math
from torch.nn import Conv2d, Dropout
from functools import reduce
from operator import mul
from einops import rearrange, repeat

import logging 
from typing import Optional, List, Tuple, Type, Any

class Template_SwinTransformer(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.switch_off = False  

    def switch_off_norms(self):
        norms = ["norm1", "norm2", "norm3"]
        norms_wts = [e + ".weight" for e in norms]
        norms_bias = [e + ".bias" for e in norms]
        for name, param in self.named_parameters():
            if param.requires_grad:
                if (name in norms_wts) or (name in norms_bias):
                    param.requires_grad = False 

    def _print_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print (" === " , name, param.data.shape)
    
    def train(self, mode=True):
        for module in self.children():
            module.train(mode)

    def swith_off_added_modules(self, ):
        self.switch_off = True 
    
    def swith_on_added_modules(self, ):
        self.switch_off = False 
        
class PatchEmbed_FIX(PatchEmbed):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, filter = [32, 64], spatial = [9, 5, 5]):
        super().__init__(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        
        self.layer_1 = nn.Conv2d(in_chans, filter[0], spatial[0], padding = spatial[0] // 2)
        self.layer_2 = nn.Conv2d(filter[0], filter[1], spatial[1], padding = spatial[1] // 2)
        self.layer_3 = nn.Conv2d(filter[1], embed_dim, patch_size, stride = patch_size)

        self.relu = nn.ReLU()

        self.proj.weight.requires_grad = False 
        self.proj.bias.requires_grad = False 
        
        self.norm.weight.requires_grad = False        
        self.norm.bias.requires_grad = False 
        
        self.use_adapters = True 

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        
        if self.use_adapters:
            # torch.Size([8, 3, 736, 1088]) --> torch.Size([8, 96, 184, 272])
            reform = self.layer_1(x)
            reform = self.relu(reform)
            reform = self.layer_2(reform)
            reform = self.relu(reform)
            reform = self.layer_3(reform)

            x = self.proj(x) + reform # B C Wh Ww
        else:
            x = self.proj(x) 
        
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x



##############################################################################
############# LORA LINEAR 

class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
        ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # will call reset_parameters via nn.Linear first 
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.bias.requires_grad = False 
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)  # (3d,) -> (3,d)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)  # (3d,)
        self.reset_parameters()
        # this will be second reset_parameters call after intial nn.Linear
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        # x -> (B,N,2d)
        # print(x.shape)
        result = x.new_zeros((*x.shape[:-1], self.out_features)) # result -> B,N,3d
        # result = x.new_zeros((self.out_features, *x.shape[1:]))
        result = result.view(-1, self.out_features)  # BN, 3d
        # result = result.view(self.out_features, -1)
        # print(result.shape)
        # print(self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )  # (BN,2d)
        # result[self.lora_ind, :] = x.reshape(
        #      self.out_features // len(self.enable_lora) * sum(self.enable_lora), -1
        # )  # (BN,2d)
        # print(result.shape)
        return result.view((*x.shape[:-1], self.out_features))

        # print(sum(x - result[self.lora_ind,:]))
        # print(result[1,:])
        # return result

    def zero_pad_weight(self,x):
        # x -> (B,N,2d)
        # print(x.shape)
        # result = x.new_zeros((*x.shape[:-1], self.out_features)) # result -> B,N,3d
        result = x.new_zeros((self.out_features, *x.shape[1:]))
        # result = result.view(-1, self.out_features)  # BN, 3d
        result = result.view(self.out_features, -1)
        # print(result.shape)
        # print(self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        # result[:, self.lora_ind] = x.reshape(
        #     -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        # )  # (BN,2d)
        result[self.lora_ind, :] = x.reshape(
            self.out_features // len(self.enable_lora) * sum(self.enable_lora), -1
        )  # (BN,2d)
        # print(result.shape)
        # return result.view((*x.shape[:-1], self.out_features))

        # print(sum(x - result[self.lora_ind,:]))
        # print(result[1,:])
        return result

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0),
                        self.lora_B.data.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data -= self.zero_pad_weight(T(delta_w * self.scaling))
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    delta_w = F.conv1d(
                        self.lora_A.data.unsqueeze(0),
                        self.lora_B.data.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).squeeze(0)
                    self.weight.data += self.zero_pad_weight(T(delta_w * self.scaling))
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),
                    self.lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
        ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.bias.requires_grad = False 
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
                                                                                                      1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class BasicLayer_Lora(BasicLayer):
    def __init__(self,
                 dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, rank=None, e2e=None):
        
        super().__init__(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, 
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
        attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, 
        use_checkpoint=False,)

        if (not e2e):
            for p in self.parameters():
                p.requires_grad = False
            
        for i in range(depth):
            # 3 means query , key, value 
            self.blocks[i].attn.qkv = MergedLinear(dim, dim * 3, r=rank, enable_lora=[True, True, True], bias=qkv_bias)
            # d_value == dim (output dim happens to be transformer dim)
            self.blocks[i].attn.proj = Linear(dim, dim, r=rank,)
           

class SwinTransformer_Lora(Template_SwinTransformer):
    def __init__(self, 
            rank=0.0, 
            embed_dim=96, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7,  mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, 
            drop_rate=0., attn_drop_rate=0.,
            use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
            patch_size=4, no_norm=None, e2e=None, **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, use_checkpoint=use_checkpoint, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, patch_size=patch_size, **kwargs)

        if (not e2e):
            self.frozen_stages = len(self.layers) + 1
            self._freeze_stages()

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        del self.layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_Lora(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False,
                rank=rank,   
                e2e=e2e,              
                )
            self.layers.append(layer)
        
        if no_norm and (not e2e):
            self.switch_off_norms()        
        self._print_trainable_params()
        
class SwinTransformer_Lora_PATCH_FIX(SwinTransformer_Lora):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, **kwargs): 
        super().__init__(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer, **kwargs)
        del self.patch_embed 
        self.patch_embed = PatchEmbed_FIX(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
    




##############################################################################
############# ADAPTERS
class AdapterModule(nn.Module):
    def __init__(self, in_feature, mid_features):
        super().__init__()
        self.proj_down = nn.Linear(in_features=in_feature, out_features=mid_features)
        self.proj_up = nn.Linear(in_features=mid_features, out_features=in_feature)
    
    def forward(self, x):
        input = x.clone()
        x = self.proj_down(x)
        x = F.relu(x)
        return self.proj_up(x) + input # Skip Connection

class SwinTransformerBlock_Adapt(SwinTransformerBlock):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_dim=-1):
        super().__init__(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            act_layer=act_layer, norm_layer=norm_layer, )
        
        for p in self.parameters():
            p.requires_grad = False
        
        self.adapter1 = AdapterModule(in_feature=dim, mid_features=dim // 4)
        self.adapter2 = AdapterModule(in_feature=dim, mid_features=dim // 4)

        
        # correct place to do near zero initialization
        self.adapter1.proj_down.weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter1.proj_down.bias.data.zero_()
        self.adapter1.proj_up.weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter1.proj_up.bias.data.zero_()

        # correct place to do near zero initialization
        self.adapter2.proj_down.weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter2.proj_down.bias.data.zero_()
        self.adapter2.proj_up.weight.data.normal_(mean=0.0, std=1e-7)
        self.adapter2.proj_up.bias.data.zero_()
        self.switch_off = None 

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        if (not self.switch_off):
            x = self.adapter1(x)
        x = shortcut + self.drop_path(x)
        
        shortcut = x
        x = self.mlp(self.norm2(x))
        if (not self.switch_off):
            x = self.adapter2(x)

        x = shortcut + self.drop_path(x)
        return x

class BasicLayer_Adapt(BasicLayer):
    def __init__(self,
                 dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, adapter_dim=None, ):
        
        super().__init__(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, 
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
        attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, 
        use_checkpoint=use_checkpoint,)
        
        for p in self.parameters():
            p.requires_grad = False
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_Adapt(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                adapter_dim=adapter_dim)
            for i in range(depth)])

# https://github.com/cs-mshah/Adapter-Bert/blob/main/adapter-bert/model/bert.py
# https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf 
class SwinTransformer_Adapter(Template_SwinTransformer):
    def __init__(self, 
            adapter_dim=0.0, 
            embed_dim=96, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7,  mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, 
            drop_rate=0., attn_drop_rate=0.,
            use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
            patch_size=4, no_norm=None, **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, use_checkpoint=use_checkpoint, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, patch_size=patch_size, **kwargs)

        self.frozen_stages = len(self.layers) + 1
        self._freeze_stages()

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        del self.layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_Adapt(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False,
                # use_checkpoint=use_checkpoint and i_layer > self.frozen_stages - 1,
                adapter_dim=adapter_dim,                
                )
            self.layers.append(layer)
        
        if no_norm:
            self.switch_off_norms()
        
        self._print_trainable_params()

    def swith_off_added_modules(self, ):
        self.switch_off = True 
        for layer in self.layers:
            for blk in layer.blocks:
                blk.switch_off = True 
                
    def swith_on_added_modules(self, ):
        self.switch_off = False 
        for layer in self.layers:
            for blk in layer.blocks:
                blk.switch_off = False  
    
class SwinTransformer_Adapter_SR(SwinTransformer_Adapter):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, **kwargs): 
        super().__init__(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer, **kwargs)
        del self.patch_embed 
        self.patch_embed = PatchEmbed_FIX(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
    



##############################################################################
############# VISUAL PROMPT RUNING 
# https://arxiv.org/pdf/2203.12119

class PromptedWindowAttention(WindowAttention):
    def __init__(
        self, num_prompts, prompt_location, dim, window_size, num_heads,
        qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., e2e=False ):
        super(PromptedWindowAttention, self).__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale,
            attn_drop, proj_drop)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

        if not e2e:
            for p in self.parameters():
                p.requires_grad = False
        self.switch_off = None 

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # account for prompt nums for relative_position_bias
        # attn: [1920, 6, 649, 649]
        # relative_position_bias: [6, 49, 49])

        if self.prompt_location == "prepend" and (not self.switch_off):
            # expand relative_position_bias
            _C, _H, _W = relative_position_bias.shape

            relative_position_bias = torch.cat((
                torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                relative_position_bias
                ), dim=1)
            relative_position_bias = torch.cat((
                torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device),
                relative_position_bias
                ), dim=-1)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            if self.prompt_location == "prepend" and (not self.switch_off):
                # expand relative_position_bias
                mask = torch.cat((
                    torch.zeros(nW, self.num_prompts, _W, device=attn.device),
                    mask), dim=1)
                mask = torch.cat((
                    torch.zeros(
                        nW, _H + self.num_prompts, self.num_prompts,
                        device=attn.device),
                    mask), dim=-1)
            # logger.info("before", attn.shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self, num_prompts, dim, 
        num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, prompt_location="prepend", e2e=False):
        super().__init__(
            dim, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer)
        
        if not e2e:
            for p in self.parameters():
                p.requires_grad = False
            
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        
        self.attn = PromptedWindowAttention(
            num_prompts, prompt_location,
            dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, e2e=e2e)
        self.switch_off = None 

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        shortcut = x
        H, W = self.H, self.W
        x = self.norm1(x)

        if self.prompt_location == "prepend" and (not self.switch_off):
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts

        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend" and (not self.switch_off):
            # expand prompts_embs
            # B, num_prompts, C --> nW*B, num_prompts, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # seperate prompt embs --> nW*B, num_prompts, C
        if self.prompt_location == "prepend" and (not self.switch_off):
            # change input size
            prompt_emb = attn_windows[:, :self.num_prompts, :]
            attn_windows = attn_windows[:, self.num_prompts:, :]
            # change prompt_embs's shape:
            # nW*B, num_prompts, C - B, num_prompts, C
            prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
            prompt_emb = prompt_emb.mean(0)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        
        x = x.view(B, H * W, C)
        # add the prompt back:
        if self.prompt_location == "prepend" and (not self.switch_off):
            x = torch.cat((prompt_emb, x), dim=1)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PromptedPatchMerging(PatchMerging):
    def __init__(self, dim, norm_layer=nn.LayerNorm, num_prompts=-1, e2e=False):
        super(PromptedPatchMerging, self).__init__(dim, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_upsampling = None
        self.switch_off = None 

        if not e2e:
            for p in self.parameters():
                p.requires_grad = False
            
    def upsample_prompt(self, prompt_emb):
        prompt_emb = torch.cat(
            (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        
        if (not self.switch_off):
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts
            prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        if (not self.switch_off):
            # add the prompt back:
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer_VPT(BasicLayer):
    def __init__(self, dim,
                 depth, 
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, 
                 num_prompts=-1, 
                 deep_prompt=None, 
                 prompt_location="prepend",
                 patch_size= -1,
                 e2e=False, 
                 **kwargs):
        super().__init__(dim=dim, depth=depth, num_heads=num_heads, 
        window_size=window_size, mlp_ratio=mlp_ratio, qk_scale=qk_scale, qkv_bias=qkv_bias, 
        drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, 
        use_checkpoint=False, **kwargs)
        

        self.num_prompts = num_prompts
        self.deep_prompt = deep_prompt
        # build blocks
        self.blocks = nn.ModuleList([
            PromptedSwinTransformerBlock(
                num_prompts=num_prompts, 
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                prompt_location=prompt_location,
                e2e=e2e)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = PromptedPatchMerging(dim=dim, norm_layer=norm_layer, num_prompts=num_prompts, e2e=e2e)
        else:
            self.downsample = None

        if not e2e:
            for p in self.parameters():
                p.requires_grad = False

        prompts = []
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + dim))  # noqa
        for i_layer in range(len(self.blocks)):
            deep_prompt_embeddings = nn.Parameter(torch.zeros( 1, num_prompts, dim))
            nn.init.uniform_(deep_prompt_embeddings.data, -val, val)
            prompts.append(deep_prompt_embeddings)
        self.deep_pompts = nn.ParameterList(prompts)
        self.switch_off = None 

    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        B = x.shape[0]  # batchsize
        num_blocks = len(self.blocks)

        for i in range(num_blocks):
            # H * W
            blk = self.blocks[i]
            blk.H, blk.W = H, W
            if (not self.switch_off):
                prompt_emb = self.deep_pompts[i].expand(B, -1, -1)
                x = torch.cat((prompt_emb, x[:, self.num_prompts:, :]), dim=1 )
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
    
class SwinTransformer_VPT(Template_SwinTransformer):
    def __init__(self, 
                prompt_DROPOUT=0.0, 
                prompt_NUM_TOKENS=20, 
                prompt_INITIATION = "random",
                prompt_MODE = "prepend",
                prompt_DEEP = False,
                no_norm=None, 

                embed_dim=96, depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24], window_size=7,  mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, 
                drop_rate=0., attn_drop_rate=0.,
                use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                patch_size=4,
                 **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, use_checkpoint=use_checkpoint, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, patch_size=patch_size, **kwargs)

        patch_size = to_2tuple(patch_size)

        self.prompt_MODE = prompt_MODE
        self.frozen_stages = len(self.layers) + 1
        self._freeze_stages()

        self.prompt_dropout = Dropout(prompt_DROPOUT)

        use_checkpoint = False 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        del self.layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_VPT(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint and i_layer > self.frozen_stages - 1,
                num_prompts=prompt_NUM_TOKENS,
                deep_prompt=prompt_DEEP,
                patch_size = patch_size, 
                )
            self.layers.append(layer)
            
        if prompt_INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))  # noqa
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_NUM_TOKENS, embed_dim))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
    
        self.prompt_DEEP = prompt_DEEP
        assert self.prompt_DEEP, "Only Deep Prompts are allowed"
        self.num_prompts = prompt_NUM_TOKENS

        if no_norm:
            self.switch_off_norms()

        self._print_trainable_params()
        self.switch_off = None 

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)        
        B = x.shape[0]
        x = rearrange(x, "B C H W  -> B (H W) C")
        
        if (not self.switch_off):
            # x = x.flatten(2).transpose(1, 2)
            prompt_embd = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
            x = torch.cat((prompt_embd, x), dim=1)

        x = self.pos_drop(x)        
        outs = []

        for i in range(self.num_layers):
            # print("===", x.shape)
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            # print(x.shape, x_out.shape)
            name = f'stage{i + 2}'
            if name in self.out_features:
                norm_layer = getattr(self, f'norm{i}')
                if (not self.switch_off):
                    x_out = x_out[:, self.num_prompts:, :]
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                # print(x_out.shape)
                outs.append(out)
        return outs

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            for module in self.children():
                module.train(False)
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    def swith_off_added_modules(self, ):
        self.switch_off = True 
        for layer in self.layers:
            layer.switch_off = True 
            if hasattr(layer, "downsample") and layer.downsample is not None :
                layer.downsample.switch_off = True
            for blk in layer.blocks:
                blk.switch_off = True 
                blk.attn.switch_off = True

    def swith_on_added_modules(self, ):
        self.switch_off = False 
        for layer in self.layers:
            layer.switch_off = False 
            if hasattr(layer, "downsample") and layer.downsample is not None :
                layer.downsample.switch_off = False 
            for blk in layer.blocks:
                blk.switch_off = False 
                blk.attn.switch_off = False 
     
class SwinTransformer_VPT_PATCH_FIX(SwinTransformer_VPT):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, **kwargs): 
        super().__init__(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer, **kwargs)
        del self.patch_embed 
        self.patch_embed = PatchEmbed_FIX(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
    

    




##############################################################################
############# LR Tokens    
class SwinTransformer_LTTKO(Template_SwinTransformer):
    def __init__(self, 
            prompt_DROPOUT=0.0, 
            prompt_NUM_TOKENS=32, no_norm=None, 
            embed_dim=96, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7,  mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, 
            drop_rate=0., attn_drop_rate=0.,
            use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
            patch_size=4, **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, use_checkpoint=use_checkpoint, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, patch_size=patch_size, **kwargs)

        self.frozen_stages = len(self.layers) + 1
        self._freeze_stages()

        patch_size = to_2tuple(patch_size)
        self.prompt_dropout = Dropout(prompt_DROPOUT)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))  # noqa

        if type(prompt_NUM_TOKENS) == list:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, embed_dim, prompt_NUM_TOKENS[0], prompt_NUM_TOKENS[0]))
        else:
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, embed_dim, prompt_NUM_TOKENS, prompt_NUM_TOKENS))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)


        # build layers
        prompts = []
        for i_layer in range(self.num_layers):
            dim=int(embed_dim * 2 ** i_layer)
            if type(prompt_NUM_TOKENS) == list:
                deep_prompt_embeddings = nn.Parameter(torch.zeros( 1, dim, prompt_NUM_TOKENS[i_layer], prompt_NUM_TOKENS[i_layer]))
            else:
                deep_prompt_embeddings = nn.Parameter(torch.zeros( 1, dim, prompt_NUM_TOKENS, prompt_NUM_TOKENS))
            nn.init.uniform_(deep_prompt_embeddings.data, -val, val)
            prompts.append(deep_prompt_embeddings)

        self.deep_pompts = nn.ParameterList(prompts)
        if no_norm:
            self.switch_off_norms()
        
        self._print_trainable_params()

    def add_tokens(self, x, Wh, Ww, prompt_embd):
        x = rearrange(x, "B (H W) C -> B C H W", H=Wh, W=Ww)
        
        prompt_embd = F.interpolate(prompt_embd, size=(Wh, Ww), mode='bicubic', align_corners=True)
        prompt_embd = self.prompt_dropout(prompt_embd)
        x += prompt_embd

        x = rearrange(x, "B C H W -> B (H W) C")
        return x

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        B = x.shape[0]
        
        if (not self.switch_off):
            prompt_embd = self.prompt_embeddings.expand(B, -1, -1, -1)
            prompt_embd = F.interpolate(prompt_embd, size=(Wh, Ww), mode='bicubic', align_corners=True)
            prompt_embd = self.prompt_dropout(prompt_embd)
            x += prompt_embd

        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]

            if (not self.switch_off):
                x = self.add_tokens(x, Wh, Ww, self.deep_pompts[i])
            
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            name = f'stage{i + 2}'
            if name in self.out_features:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def train(self, mode=True):
        for module in self.children():
            module.train(mode)    
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            self.prompt_dropout.train()
        
class SwinTransformer_LTTKO_PATCH_FIX(SwinTransformer_LTTKO):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm, **kwargs): 
        super().__init__(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer, **kwargs)
        del self.patch_embed 
        self.patch_embed = PatchEmbed_FIX(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
    

    def swith_off_added_modules(self, ):
        self.switch_off = True 
        self.patch_embed.use_adapters = False 
        
    def swith_on_added_modules(self, ):
        self.switch_off = False 
        self.patch_embed.use_adapters = True 
        





##############################################################################
############# Sparicification 
# https://arxiv.org/pdf/2111.09805    

class SwinTransformerBlock_Sparse(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        
        x = shortcut + self.drop_path(x)
        x_old =  x
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, x_old

class BasicLayer_Sparse(BasicLayer):
    def __init__(self,
                 dim, depth, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint)
                 
        
        new_blocks = nn.ModuleList([])
        for i in range(depth):
            if i != depth-1 :
                new_blocks.append( self.blocks[i] )
            else:
                block = SwinTransformerBlock_Sparse(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer)
                new_blocks.append(block)
        del self.blocks
        self.blocks = new_blocks
        
        
    def forward(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            import pdb
            pdb.set_trace()
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
        
class SwinTransformer_Sparse(SwinTransformer):
    def __init__(self,  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, 
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
        drop_path_rate=0.2, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, 
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,  
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)
        
        assert self.layers[-1].downsample == None

        self.drop_feats = nn.Dropout(0.9)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i == self.num_layers - 1:
                # print(x_out.mean(), x_out.mean(-1)[:,:100])
                # print(x_out.mean())
                x_out = self.drop_feats.train()(x_out)
                # print(x_out.mean())
                


            name = f'stage{i + 2}'
            if name in self.out_features:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

class SwinTransformer_Sparse2(SwinTransformer):
    def __init__(self,  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, 
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
        drop_path_rate=0.2, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, 
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,  
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, **kwargs)
        
        assert self.layers[-1].downsample == None

        # build layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        layers_new = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer != self.num_layers -1:
                layer = self.layers[i_layer]
                layers_new.append(layer)
            else:
                layer = BasicLayer_Sparse(
                    dim=int(embed_dim * 2 ** i_layer),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=False)
                layers_new.append(layer)
        del self.layers
        self.layers = layers_new
        self.layers[-1].blocks
        
        
    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i == self.num_layers - 1:
                # print(x_out.mean(), x_out.mean(-1)[:,:100])
                # print(x_out.mean())
                
                import pdb
                pdb.set_trace()
                x_out = self.drop_feats.train()(x_out)
                # print(x_out.mean())
                


            name = f'stage{i + 2}'
            if name in self.out_features:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

        
        
    



##############################################################################
############# FAN Network
# https://github.com/NVlabs/FAN/blob/master/models/fan.py#L510
# FANBlock(nn.Module)
# https://arxiv.org/pdf/2204.12451    

class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., linear=False, drop_path=0., 
                 mlp_hidden_dim=None, act_layer=nn.GELU, drop=None, norm_layer=nn.LayerNorm, cha_sr_ratio=1, c_head_num=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim//self.cha_sr_ratio, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # (in_features=dim
        self.norm_v = norm_layer(dim//self.cha_sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.apply(self._init_weights)
        self.switch_off = False 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1,-2)
        _, _, N, _  = k.shape

        # print(k.shape, k.max(), k.min())
        # k = k / (k.size(-1) ** 0.5)
        k = k.softmax(dim=-2).mean(dim=-1, keepdim=True)
        
        # k = k.to(dtype=torch.float16)
        # k = k / (k.size(-1) ** 0.5)
        # k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))
        # k = k.float()
        
        attn = torch.nn.functional.sigmoid(q @ k)
        return attn  * self.temperature

    def forward(self, x):
        if not self.switch_off:
            B, N, C = x.shape
            v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

            attn = self._gen_attn(q, k)
            attn = self.attn_drop(attn)

            Bv, Hd, Nv, Cv = v.shape
            v = self.norm_v(self.mlp(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv))).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)

            repeat_time = N // attn.shape[-1]
            attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
            x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
            # Channel attention 
            # (attn * v.transpose(-1, -2)).transpose(-1, -2) #attn 
            return x
        else:
            return self.mlp(x)        
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class SwinTransformerBlock_FAN(SwinTransformerBlock):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, e2e=False, eta=1.0, switch_off_mlp=True):
        
        super().__init__(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
        act_layer=act_layer, norm_layer=norm_layer)
        
        if not e2e:
            for p in self.parameters():
                p.requires_grad = False
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ChannelProcessing(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            drop_path=drop_path, drop=drop, mlp_hidden_dim = mlp_hidden_dim, c_head_num = None) 

        if switch_off_mlp and not e2e:
            for p in self.mlp.mlp.parameters():
                p.requires_grad = False


        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        
    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        ####################################################################################
        # NEW FAN CODE 
        ####################################################################################
        # FFN
        x = shortcut + self.drop_path( self.gamma1 *  x)
        x = x + self.drop_path(self.gamma2 *  self.mlp(self.norm2(x)))

        return x

class BasicLayer_FAN(BasicLayer):
    def __init__(self,
                 dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, switch_off_mlp=True, e2e=False ):
        
        super().__init__(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, 
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, 
        attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, 
        use_checkpoint=use_checkpoint,)
        
        if not e2e:
            for p in self.parameters():
                p.requires_grad = False
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_FAN(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                e2e=e2e, 
                switch_off_mlp=switch_off_mlp, )
            for i in range(depth)])

        # for i in range(depth):
        #     mlp_hidden_dim=int(dim * mlp_ratio)
        #     self.blocks[i].mlp = ChannelProcessing(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, 
        #                                 drop_path=drop_path[i], drop=drop, mlp_hidden_dim = mlp_hidden_dim, c_head_num = None)
        #     if switch_off_mlp and not e2e:
        #         for p in self.blocks[i].mlp.mlp.parameters():
        #             p.requires_grad = False
    
class SwinTransformer_FAN(Template_SwinTransformer):
    def __init__(self, 
            no_norm=None, 
            embed_dim=96, depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24], window_size=7,  mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, 
            drop_rate=0., attn_drop_rate=0.,
            use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
            patch_size=4, switch_off_mlp=True, e2e=False, **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, use_checkpoint=use_checkpoint, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, patch_size=patch_size, **kwargs)

        if not e2e:
            self.frozen_stages = len(self.layers) + 1
            self._freeze_stages()

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        del self.layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_FAN(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False,
                switch_off_mlp=switch_off_mlp, 
                e2e=e2e, 
                )
            self.layers.append(layer)
        
        if no_norm and not e2e:
            self.switch_off_norms()
        
        self._print_trainable_params()

    def swith_off_added_modules(self, ):
        self.switch_off = True 
        for layer in self.layers:
            for blk in layer.blocks:
                blk.mlp.switch_off = True 
                
    def swith_on_added_modules(self, ):
        self.switch_off = False 
        for layer in self.layers:
            for blk in layer.blocks:
                blk.mlp.switch_off = False  
        

##############################################################################
############# Register 
# https://arxiv.org/pdf/2309.16588
class BasicLayer_REGISTERS(BasicLayer):
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, 
                 num_prompts=1, e2e=False,  prompt_location="prepend", **kwargs):
        super().__init__(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size, 
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint, **kwargs)

        self.num_prompts = num_prompts
        
        self.blocks = nn.ModuleList([
            PromptedSwinTransformerBlock(
                num_prompts=num_prompts, 
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                prompt_location=prompt_location,
                e2e=e2e)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = PromptedPatchMerging(dim=dim, norm_layer=norm_layer, num_prompts=num_prompts, e2e=e2e)
        else:
            self.downsample = None


        if not e2e:
            for p in self.parameters():
                p.requires_grad = False

        self.switch_off = None 
        
    def forward(self, x, H, W):
        REGISTERS = None 
        if not self.switch_off:
            REGISTERS = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        B = x.shape[0]  # batchsize
        num_blocks = len(self.blocks)

        if not self.switch_off:
            x = torch.cat((REGISTERS, x), dim=1)

        for i in range(num_blocks):        
            blk = self.blocks[i]
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
            
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W
    
class SwinTransformer_Register(Template_SwinTransformer):
    def __init__(self, 
                prompt_DROPOUT=0.0, 
                prompt_NUM_TOKENS=20, 
                prompt_INITIATION = "random",
                prompt_MODE = "prepend",
                no_norm=None, 
                embed_dim=96, depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24], window_size=7,  mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, 
                drop_rate=0., attn_drop_rate=0.,
                use_checkpoint=False, drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                patch_size=4,
                e2e=False, 
                 **kwargs):
        super().__init__(embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, use_checkpoint=use_checkpoint, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, patch_size=patch_size, **kwargs)

        patch_size = to_2tuple(patch_size)

        if not e2e:
            self.frozen_stages = len(self.layers) + 1
            self._freeze_stages()

        self.prompt_dropout = Dropout(prompt_DROPOUT)

        use_checkpoint = False 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        del self.layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_REGISTERS(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint and i_layer > self.frozen_stages - 1,
                num_prompts=prompt_NUM_TOKENS, 
                e2e=e2e, 
                )
            self.layers.append(layer)
                
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, prompt_NUM_TOKENS, embed_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
    
        self.num_prompts = prompt_NUM_TOKENS

        if no_norm and not e2e:
            self.switch_off_norms()

        self._print_trainable_params()
        self.switch_off = None 

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)        
        B = x.shape[0]
        x = rearrange(x, "B C H W  -> B (H W) C")

        # print("=====", x.shape)
        if (not self.switch_off):
            prompt_embd = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
            x = torch.cat((prompt_embd, x), dim=1)

        x = self.pos_drop(x)        
        outs = []

        for i in range(self.num_layers):
            # print("===", x.shape)
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            name = f'stage{i + 2}'
            if name in self.out_features:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                if (not self.switch_off):
                    x_out = x_out[:, self.num_prompts:, :]
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            for module in self.children():
                module.train(False)
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    
    def swith_off_added_modules(self, ):
        self.switch_off = True 
        for layer in self.layers:
            layer.switch_off = True 
            if hasattr(layer, "downsample") and layer.downsample is not None :
                layer.downsample.switch_off = True
            for blk in layer.blocks:
                blk.switch_off = True 
                blk.attn.switch_off = True

    def swith_on_added_modules(self, ):
        self.switch_off = False 
        for layer in self.layers:
            layer.switch_off = False 
            if hasattr(layer, "downsample") and layer.downsample is not None :
                layer.downsample.switch_off = False 
            for blk in layer.blocks:
                blk.switch_off = False 
                blk.attn.switch_off = False 
     


##############################################################################
############# MiNT 
class SwinTransformer_Mint(Template_SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for name, param in self.named_parameters():
            if "norm" in name:continue 
            param.requires_grad = False
        self._print_trainable_params()

    
    
##############################################################################
############# RobustSAM
# https://github.com/robustsam/RobustSAM/tree/main/robust_segment_anything/modeling

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SKDown(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SKDown, self).__init__()
        self.maxpool_conv = nn.Sequential(
            SelectiveConv(kernel_size, padding, bias, reduction, in_channels, out_channels, first=first)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class DNCBlock_combined(nn.Module):
    def __init__(self, vit_dim):
        super(DNCBlock_combined, self).__init__()
        self.num_channels = vit_dim
        self.channel_attention = CABlock(2*self.num_channels)
        self.SEMBlock = SKDown(3, 1, False, 16, self.num_channels, self.num_channels, first=False)

    def forward(self, x):
        x_in = self.SEMBlock(x)
        x_all = torch.cat([x, x_in], dim=1)
        output = self.channel_attention(x_all)
        return output   

class CABlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CABlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = self.squeeze(x).view(batch_size, channels)
        excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)
        return x * excitation

class SelectiveConv(nn.Module):
    def __init__(self, kernel_size, padding, bias, reduction, in_channels, out_channels, first=False):
        super(SelectiveConv, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.selector = Selector(out_channels, reduction=reduction)
        self.IN = nn.InstanceNorm2d(in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):    
        if self.first:
            f_input = x
            s_input = x
        else:
            f_input = self.BN(x.clone())
            f_input = self.relu(f_input)

            s_input = self.IN(x.clone())
            s_input = self.relu(s_input)

        out1 = self.conv1(f_input)
        out2 = self.conv2(s_input)

        out = out1 + out2

        att1, att2 = self.selector(out)
        out = torch.mul(out1, att1) + torch.mul(out2, att2)

        return out    
    
class Selector(nn.Module):
    def __init__(self, channel, reduction=16, crp_classify=False):
        super(Selector, self).__init__()
        self.spatial_attention = 4
        self.in_channel = channel * (self.spatial_attention ** 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.spatial_attention, self.spatial_attention))

        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
        )
        self.att_conv1 = nn.Linear(self.in_channel // reduction, self.in_channel)
        self.att_conv2 = nn.Linear(self.in_channel // reduction, self.in_channel)

    def forward(self, x):
        b, c, H, W = x.size()
        y = self.avg_pool(x).reshape(b, -1)
        y = self.fc(y)
        att1 = self.att_conv1(y).view(b, c, self.spatial_attention, self.spatial_attention)
        att2 = self.att_conv2(y).view(b, c, self.spatial_attention, self.spatial_attention)
        attention = torch.stack((att1, att2))
        attention = nn.Softmax(dim=0)(attention)
        att1 = F.interpolate(attention[0], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")
        att2 = F.interpolate(attention[1], scale_factor=(H / self.spatial_attention, W / self.spatial_attention), mode="nearest")
        return att1, att2

class MaskFeatureBlock(nn.Module):
    def __init__(self, transformer_dim):
        super(MaskFeatureBlock, self).__init__()        
        self.dnc_block_combined =  DNCBlock_combined(transformer_dim) 
        self.fgm_block = FGMBlock(transformer_dim )  
        self.conv_layer = nn.Conv2d(2 * transformer_dim, transformer_dim, kernel_size=3, padding=1)        

        self.downsample_layer = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim // 4, 3, 1, 1), 
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim, 3, 1, 1)
        )

    def forward(self, x, clear=True):
        input_shape = x.shape
        if not clear:
            x = self.dnc_block_combined(x)
            x = self.fgm_block(x)                
            x = self.conv_layer(x)   
        output = self.downsample_layer(x)
        assert input_shape == output.shape
        return output

class FGMBlock(nn.Module):
    def __init__(self, vit_dim):
        super(FGMBlock, self).__init__()
        self.num_channels = 2 * vit_dim
        self.conv_layer = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=1)

    def forward(self, x):
        fft_map = torch.fft.fft2(x, dim=(-2, -1))
        magnitude_map = torch.abs(fft_map)
        phase_map = torch.angle(fft_map)
        
        modified_magnitude = self.conv_layer(magnitude_map)
        
        real_part = modified_magnitude * torch.cos(phase_map)
        imag_part = modified_magnitude * torch.sin(phase_map)
        modified_fft_map = torch.complex(real_part, imag_part)
        reconstructed_x = torch.real(torch.fft.ifft2(modified_fft_map, dim=(-2, -1)))
        return reconstructed_x

class FirstLayerFeatureBlock(nn.Module):
    def __init__(self, vit_dim, transformer_dim, factor=32):
        super(FirstLayerFeatureBlock, self).__init__()
        self.dnc_block_combined =  DNCBlock_combined(transformer_dim) 
        self.fgm_block = FGMBlock(transformer_dim)  
        self.conv_layer = nn.Conv2d(2*transformer_dim, transformer_dim, kernel_size=3, padding=1)

        if factor == 32:
            self.downsample_layer = nn.Sequential(
                # nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                nn.Conv2d(vit_dim, transformer_dim, kernel_size=4, stride=4),
                LayerNorm2d(transformer_dim), nn.GELU(), 
                # nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2)
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=4, stride=4),
                LayerNorm2d(transformer_dim), nn.GELU(), 
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=2, stride=2, padding=1),
            )
        elif factor == 2:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, stride=1, padding=1),
                LayerNorm2d(transformer_dim), nn.GELU(), 
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),)
        elif factor == 4:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim), nn.GELU(), 
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=2, stride=2))
        elif factor == 8:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=4, stride=4),
                LayerNorm2d(transformer_dim),
                nn.GELU(), 
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=2, stride=2))
        elif factor == 16:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=4, stride=4),
                LayerNorm2d(transformer_dim),
                nn.GELU(), 
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=4, stride=4, padding=1))

        self.upscale_layer = nn.Sequential(
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=1, stride=1),
                LayerNorm2d(transformer_dim),
                nn.GELU(), 
                nn.Conv2d(transformer_dim, transformer_dim, kernel_size=1, stride=1))

    def forward(self, x, clear=True):
        x = self.downsample_layer(x)
        if not clear:
            # print(x.mean().item(), x.max().item(), x.min().item())
            x = self.dnc_block_combined(x)
            # print(x.mean().item() , x.max().item(), x.min().item())
            x = self.fgm_block(x)                
            # print(x.mean().item(), x.max().item(), x.min().item())
            x = self.conv_layer(x)   

        output = self.upscale_layer(x)
        return output

class LastLayerFeatureBlock(nn.Module):
    def __init__(self, transformer_dim):
        super(LastLayerFeatureBlock, self).__init__()

        self.layer_norm = LayerNorm2d(transformer_dim) 
        self.dnc_block_combined =  DNCBlock_combined(transformer_dim) 
        self.fgm_block = FGMBlock(transformer_dim)  
        self.conv_layer = nn.Conv2d(2*transformer_dim, transformer_dim, kernel_size=3, padding=1)             
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(), 
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=2, stride=2)
        )

    def forward(self, x, clear=True):
        # print(x.shape, x.mean().item(), x.max().item(), x.min().item())
        x = self.layer_norm(x)
        input_shape = x.shape
        if not clear:
            x = self.dnc_block_combined(x)
            # print(x.shape, x.mean().item(), x.max().item(), x.min().item())
            x = self.fgm_block(x)                
            # print(x.shape, x.mean().item(), x.max().item(), x.min().item())
            x = self.conv_layer(x)   

        output = self.upsample_layer(x)
        assert input_shape == output.shape
        return output

class TokenBlock(nn.Module):
    def __init__(self, input_dim, split_dim=3, mlp_dim=None):
        super(TokenBlock, self).__init__()
        self.input_dim = input_dim
        # breakpoint()
        self.split = split_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim)
        )
        
        self.IN_layer_I = nn.InstanceNorm1d(input_dim)
        self.IN_layer_II = nn.InstanceNorm1d(input_dim)

    def forward(self, x, mlp=True):
        x = self.IN_layer_I(x)
        x = self.IN_layer_II(x)
        x = rearrange(x, "B C (S D) -> B C S D" , S=self.split)
        output = self.mlp(x)
        output = rearrange(output, "B C S D -> B C (S D)")
        return output    

class SwinTransformer_RobustSAM(Template_SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frozen_stages = len(self.layers) + 1
        self._freeze_stages()
        self.switch_off_norms()
        self._print_trainable_params()



##############################################################################
############# Cross exchange (N-local network) features accross layers (Robust Onion)

class NN_down_layer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, max_pool_enabled=True, scale=1, linear=True, add_norm=None):
        super().__init__()
        self.max_pool_enabled = max_pool_enabled
        self.linear = linear
        self.kernel = kernel
        if linear:
            self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        if max_pool_enabled:
            self.max_pool = nn.MaxPool2d(kernel_size=(kernel))
        self.scale = scale
        # for name, param in self.named_parameters():
        #     param.requires_grad = False 
        self.apply(self.init_weights)
        self.add_norm = add_norm
        if add_norm:
            self.layer_norm1 = nn.LayerNorm(in_dim)
            self.layer_norm2 = nn.LayerNorm(out_dim)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        if self.max_pool_enabled:
            x = self.max_pool(x)
        
        new_size = x.shape[2:]
        if self.add_norm:
            x = rearrange(x , "B C H W -> B (H W) C")
            x = self.layer_norm1(x)
            x = rearrange(x , "B (H W) C -> B C H W", H=new_size[0], W=new_size[1])
            
        if self.linear:
            x = rearrange(x , "B C H W -> B (H W) C")
            x = self.mlp(x)
        else:
            x = self.conv(x)
            x = rearrange(x , "B C H W -> B (H W) C")
        
        if self.add_norm:
            x = self.layer_norm2(x)
            
        x = x * self.scale 

        return x, new_size

class NN_UPscale_layer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, max_pool_enabled=True, scale=1, linear=True, add_norm=None):
        super().__init__()
        self.max_pool_enabled = max_pool_enabled
        if max_pool_enabled:
            self.pixel_shuffle_layer = nn.PixelShuffle(kernel)

        self.linear = linear
        if linear:
            self.mlp = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.zeros_(self.mlp.weight)
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=False)
            nn.init.zeros_(self.conv.weight)
        
        self.add_norm = add_norm
        if add_norm:
            self.layer_norm1 = nn.LayerNorm(in_dim)
            self.layer_norm2 = nn.LayerNorm(out_dim)
            

    def forward(self, x, target_size, curr_size, 
        mode='bicubic' #'bilinear'
        ):
        H,W = curr_size
        x = rearrange(x , "B (H W) C -> B C H W", H=H, W=W)
        if self.max_pool_enabled:
            x = self.pixel_shuffle_layer(x)
        if x.shape[2:] != target_size:
            x = F.interpolate(x, target_size, mode=mode)
        
        H,W = target_size
        if self.add_norm:
            x = rearrange(x , "B C H W -> B (H W) C")
            x  = self.layer_norm1(x)
            x = rearrange(x , "B (H W) C -> B C H W", H=H, W=W)

        if self.linear:
            x = rearrange(x , "B C H W -> B (H W) C")
            x = self.mlp(x)
            x = rearrange(x , "B (H W) C -> B C H W", H=H, W=W)
        else:
            x = self.conv(x)
        
        if self.add_norm:
            x = rearrange(x , "B C H W -> B (H W) C")
            x  = self.layer_norm2(x)
            x = rearrange(x , "B (H W) C -> B C H W", H=H, W=W)

        return x

class Non_local(nn.Module):
    def __init__(self, linear_version=False, kernel=3, num_layers=1, embed_dim=96, add_norm=None):
        super().__init__()

        self.Query = nn.ModuleList()
        self.Key = nn.ModuleList()
        self.Value = nn.ModuleList()
        self.Out = nn.ModuleList()
        self.linear_version = linear_version
        self.kernel = kernel
        self.num_layers = num_layers
        
        for i_layer in range(num_layers):
            dim=int(embed_dim * 2 ** i_layer)   
            scale = dim ** -0.5
            self.Query.append( NN_down_layer(dim, embed_dim // 2, scale=scale, linear=self.linear_version, kernel=self.kernel, add_norm=add_norm) )
            self.Key.append( NN_down_layer(dim, embed_dim // 2, linear=self.linear_version, kernel=self.kernel, add_norm=add_norm ) )
            self.Value.append( NN_down_layer(dim, self.kernel * self.kernel * (embed_dim // 2), linear=self.linear_version, kernel=self.kernel, add_norm=add_norm) )
            # self.Out[0] is ignored in enhancer 
            if i_layer != 0 :
                self.Out.append( NN_UPscale_layer(embed_dim // 2, dim, linear=self.linear_version, kernel=self.kernel, add_norm=add_norm) )
            else:
                self.Out.append( nn.Identity() )
        # self.Out[0].conv.weight.requires_grad = False 
        
    def convert_to_KQV(self, y, i_layer):
        query_y, new_size = self.Query[i_layer](y)
        key_y, _ = self.Key[i_layer](y)
        value_y, _ = self.Value[i_layer](y)
        return query_y, key_y, value_y, new_size

    def compute_attention(self, Query, Key, Value):
        Query = torch.cat(Query, 1)
        Key = torch.cat(Key, 1)
        Value = torch.cat(Value, 1)
        
        Atten = Query @ Key.permute(0, 2,1)
        Atten = Atten.softmax(-1)             
        y = (Atten @ Value)
        return y 

    def forward(self, outs):
        Query = []
        Key = []
        Value = []
        spatial_res = []
        new_spatial_res = []
        for i in range(self.num_layers):
            out = outs[i]
            H,W = out.shape[2:]
            spatial_res.append( (H,W) )

            query_y, key_y, value_y, new_size = self.convert_to_KQV(out, i)
            new_spatial_res.append(new_size)
            Query.append(query_y)
            Key.append(key_y)
            Value.append(value_y)
        out = self.compute_attention(Query, Key, Value)
        curr = 0 
        for i_layer in range(self.num_layers):
            spatial_dim = reduce( mul, new_spatial_res[i_layer])
            if i_layer != 0:
                output = self.Out[i_layer](
                    out[:, curr : curr + spatial_dim, :], 
                    target_size=spatial_res[i_layer], 
                    curr_size=new_spatial_res[i_layer]
                )
                outs[i_layer] = (outs[i_layer] + output).contiguous()  
            curr += spatial_dim
        return outs

class MHSA(nn.Module):
    def __init__(
        self,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        embed_dim=0, 
        num_layers = -1, 
        kernel = 3, 
        drop_path=0,
        layer_scale_init_value: float = 1e-5,
        layer_norm = nn.LayerNorm, 
        ) -> None:
        
        super().__init__()

        torch.autograd.set_detect_anomaly(True)


        self.num_heads = num_heads

        self.QKV = nn.ModuleList()
        self.out = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.norms_input = nn.ModuleList()
        self.layer_scaling = nn.ParameterList()
        self.up_scaling = nn.ModuleList()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_layers  = num_layers

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mode='bilinear'
        
        for i_layer in range(num_layers):
            dim=int(embed_dim * 2 ** i_layer)   
            scale = dim ** -0.5
            self.pools.append(nn.MaxPool2d(kernel_size=(kernel)))
            qkv = nn.Linear(dim, (self.head_dim * num_heads) * 3, bias=qkv_bias)
            self.QKV.append(qkv)
            if i_layer != 0 :
                self.out.append( nn.Linear( (self.head_dim * num_heads) , dim) )
                
                nn.init.zeros_(self.out[-1].weight)
                nn.init.zeros_(self.out[-1].bias)

                upsample_layer = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=3, padding=0)
                self.up_scaling.append(upsample_layer)
                self.layer_scaling.append(nn.Parameter( layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True ) )
            else:
                self.out.append( nn.Identity() )
                self.up_scaling.append(  nn.Identity() )
                self.layer_scaling.append(nn.Parameter( torch.zeros((dim, 1, 1)), requires_grad=False ) )
            self.norms_input.append( layer_norm(dim) )


    def compute_qkv(self, x_local, i_layer):
        in_shape = x_local.shape          
        in_shape = in_shape[2:]
        # print(in_shape)
        x_local = rearrange(x_local, 'B C H W -> B (H W) C').contiguous()
        x_local = self.norms_input[i_layer]( x_local )
        x_local = rearrange(x_local, 'B (H W) C -> B C H W', H=in_shape[0], W=in_shape[1]).contiguous()
        x_local = self.pools[i_layer] (x_local)
        out_shape = x_local.shape
        out_shape = out_shape[2:]

        x_local = torch.flatten(x_local, start_dim=2).transpose(-2, -1)  # (B, N, C)
        B, N = x_local.shape[:2]
        qkv = self.QKV[i_layer](x_local)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # print(qkv.shape)
        return qkv, in_shape, out_shape
        
    def compute_atten(self, Vals, B, total_spatial_tokens):
        q, k, v = Vals.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # print(q.shape, k.shape, v.shape)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(1, 2).reshape(B, total_spatial_tokens, -1)
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Vals = []
        inshapes = []
        outshapes = []

        B = x[0].shape[0]
        for i_layer in range(self.num_layers):
            x_local = x[i_layer]
            qkv, in_shape, out_shape = self.compute_qkv( x_local, i_layer)
            inshapes.append( in_shape )  
            outshapes.append( out_shape )
            Vals.append(qkv)
        
        Vals = torch.cat(Vals, 3)
        atten = self.compute_atten( Vals, B, sum([reduce( mul, e ) for e in outshapes]) )
         
        curr = 0 
        for i_layer in range(self.num_layers):
            spatial_dim = reduce( mul, outshapes[i_layer])
            out_local = atten[:, curr : curr + spatial_dim, :]
            curr += spatial_dim

            out_local = self.out[i_layer](out_local)
            out_local = self.proj_drop(out_local)
            out_local = out_local.transpose(-2, -1).reshape(
                B, -1, outshapes[i_layer][0], outshapes[i_layer][1]
            )

            out_local = self.up_scaling[i_layer](out_local)
            # print(reduce( mul, inshapes[i_layer]) - spatial_dim)
            out_local = F.interpolate(out_local, inshapes[i_layer], mode=self.mode)
            x[i_layer] += self.drop_path( self.layer_scaling[i_layer] *  out_local )

        return x




class SwinTransformer_NN(Template_SwinTransformer):
    def __init__(self, no_norm=None, embed_dim=96, depths=[2, 2, 6, 2], linear_version=False, add_norm=False, **kwargs): 
        super().__init__(embed_dim=embed_dim, depths=depths, **kwargs)
 
        self.frozen_stages = len(self.layers) + 1
        self._freeze_stages()
        if no_norm:
            self.switch_off_norms()
    
        self.nn_block = Non_local(linear_version=linear_version, kernel=3, 
            num_layers=self.num_layers, embed_dim=embed_dim, add_norm=add_norm)
        
        # self.nn_block.Out[1].mlp.weight
        # self.nn_block.Query[0].mlp.weight
        # self.nn_block.Key[0].mlp.weight
        # self.nn_block.Value[0].mlp.weight
        self._print_trainable_params()
    
    def forward(self, x):
        outs = super().forward(x)
        # [e.shape for e in outs]
        # print([e.mean() for e in outs])
        if (not self.switch_off):    
            outs = self.nn_block(outs)
        # print([e.mean() for e in outs])
        return outs

class SwinTransformer_NN_LTTKO(SwinTransformer_LTTKO):
    def __init__(self, embed_dim=96, depths=[2, 2, 6, 2], linear_version=False, add_norm=False, mhsa=None, **kwargs): 
        super().__init__(embed_dim=embed_dim, depths=depths, **kwargs) 
        if mhsa:
            self.nn_block = MHSA(num_layers=self.num_layers, embed_dim=embed_dim)
        else:
            self.nn_block = Non_local(linear_version=linear_version, kernel=3, 
                num_layers=self.num_layers, embed_dim=embed_dim, add_norm=add_norm, )
        self._print_trainable_params()

    def forward(self, x):
        outs = super().forward(x)
        # print([e.mean() for e in outs])
        if (not self.switch_off):    
            outs = self.nn_block(outs)
            # outs = self.nn_block([e.detach() for e in outs])
        # print([e.mean() for e in outs])
        return outs

class SwinTransformer_MHSA(Template_SwinTransformer):
    def __init__(self, no_norm=None, embed_dim=96, depths=[2, 2, 6, 2], **kwargs): 
        super().__init__(embed_dim=embed_dim, depths=depths, **kwargs)
        self.frozen_stages = len(self.layers) + 1
        self._freeze_stages()
        # self.layers[0]
        if no_norm:
            self.switch_off_norms()
        self.nn_block = MHSA(num_layers=self.num_layers, embed_dim=embed_dim)
        self._print_trainable_params()
    
    def forward(self, x):
        outs = super().forward(x)
        if (not self.switch_off):    
            outs = self.nn_block(outs)
        print([torch.isnan(e).any().item() for e in outs])
        return outs

    



##### Baselines
def build_swint_backbone_vpt(cfg, prompt_DEEP=None):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ VISUAL PROMPT TUNNING  ****** \n\n")
    return SwinTransformer_VPT(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        prompt_DEEP=prompt_DEEP,
        no_norm=cfg.MODEL.NO_NORM,
    )

def build_swint_backbone_lr_tko(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ LR TOKens (Tk0) ****** \n\n")
    if cfg.MODEL.BACKBONE.PROMPT_NUM_TOKEN:
        prompt_NUM_TOKENS =  cfg.MODEL.BACKBONE.PROMPT_NUM_TOKEN
    else:
        prompt_NUM_TOKENS = 32

    return SwinTransformer_LTTKO(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        no_norm=cfg.MODEL.NO_NORM,
        prompt_NUM_TOKENS=prompt_NUM_TOKENS,
    )

def build_swint_backbone_adapter(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ ADAPTER LAYERS ****** \n\n")
    return SwinTransformer_Adapter(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        adapter_dim=None,
        no_norm=cfg.MODEL.NO_NORM,
    )

def build_swint_backbone_lora(cfg, lora_rank=1):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ LORA  ****** \n\n")
    return SwinTransformer_Lora(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        rank=lora_rank,
        no_norm=cfg.MODEL.NO_NORM,
        e2e=cfg.MODEL.BACKBONE_E2E,
    )



##### Previous Work 
def build_swint_backbone_fan(cfg, switch_off_mlp=True, e2e=False):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ FAN channel attention ****** \n\n")
    return SwinTransformer_FAN(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        no_norm=cfg.MODEL.NO_NORM,
        switch_off_mlp=switch_off_mlp,
        e2e=e2e,
    )

def build_swint_backbone_register(cfg, e2e=False ):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ Registors (non trainable)****** \n\n")
    return SwinTransformer_Register(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        no_norm=cfg.MODEL.NO_NORM,
        e2e=e2e, 
    )

def build_swint_backbone_mint(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ MInt (only layer norm) ****** \n\n")
    return SwinTransformer_Mint(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
    )

def build_swint_backbone_RobustSAM(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ RobustSAM ****** \n\n")
    return SwinTransformer_RobustSAM(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
    )

def build_swint_backbone_NN(cfg, linear_version=False, backbone_type=None, add_norm=False, mhsa=None):
    logger = logging.getLogger(__name__)
    logger.info(f"\n\n  ****** W/ Non-Local Network :: Linear : {linear_version} :: Back: {backbone_type} ****** \n\n")
    parms = dict(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        no_norm=cfg.MODEL.NO_NORM,
        linear_version=linear_version,
        add_norm=add_norm, 
    )
    
    if backbone_type:
        if cfg.MODEL.BACKBONE.PROMPT_NUM_TOKEN:
            parms['prompt_NUM_TOKENS'] =  cfg.MODEL.BACKBONE.PROMPT_NUM_TOKEN
        else:
            parms['prompt_NUM_TOKENS'] = 32
        return SwinTransformer_NN_LTTKO(mhsa=mhsa, **parms)
    elif mhsa:
        del parms['linear_version']
        del parms['add_norm']
        return SwinTransformer_MHSA(**parms)
    else:
        # return SwinTransformer_NN(
        return SwinTransformer_NN(**parms)




##### SR
def build_swint_backbone_lr_tko_patchfix(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ LR TOKens (Tk0) + SR Patch ****** \n\n")
    return SwinTransformer_LTTKO_PATCH_FIX(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        no_norm=cfg.MODEL.NO_NORM,
    )

##### SR
def build_swint_backbone_lora_patchfix(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ LR TOKens (Tk0) + SR Patch ****** \n\n")
    return SwinTransformer_Lora_PATCH_FIX(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        no_norm=cfg.MODEL.NO_NORM,
    )

##### SR
def build_swint_backbone_vpt_patchfix(cfg, prompt_DEEP=None):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ VISUAL PROMPT TUNNING + SR ****** \n\n")
    return SwinTransformer_VPT_PATCH_FIX(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        prompt_DEEP=prompt_DEEP,
        no_norm=cfg.MODEL.NO_NORM,
    )

##### SR
def build_swint_backbone_adapter_patchfix(cfg):
    logger = logging.getLogger(__name__)
    logger.info("\n\n  ****** W/ ADAPTER LAYERS + SR ****** \n\n")
    return SwinTransformer_Adapter_SR(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES,
        adapter_dim=None,
        no_norm=cfg.MODEL.NO_NORM,
    )




def build_swint_backbone_sparse(cfg):
    return SwinTransformer_Sparse(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES
    )

def build_swint_backbone_sparse2(cfg):
    return SwinTransformer_Sparse2(
        patch_size=4,
        in_chans=3,
        embed_dim=cfg.MODEL.SWINT.EMBED_DIM,
        depths=cfg.MODEL.SWINT.DEPTHS,
        num_heads=cfg.MODEL.SWINT.NUM_HEADS,
        window_size=cfg.MODEL.SWINT.WINDOW_SIZE,
        mlp_ratio=cfg.MODEL.SWINT.MLP_RATIO,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=cfg.MODEL.SWINT.DROP_PATH_RATE,
        norm_layer=nn.LayerNorm,
        ape=cfg.MODEL.SWINT.APE,
        patch_norm=True,
        frozen_stages=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        backbone_arch=cfg.MODEL.BACKBONE.CONV_BODY,
        use_checkpoint=cfg.MODEL.BACKBONE.USE_CHECKPOINT,
        out_features=cfg.MODEL.BACKBONE.OUT_FEATURES
    )


