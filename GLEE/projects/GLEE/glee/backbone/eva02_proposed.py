
from detectron2.modeling import BACKBONE_REGISTRY
from .eva02 import SimpleFeaturePyramid, logger, EVA02_ViT, \
    ShapeSpec, get_abs_pos, Block, Attention, xops

from functools import partial



from operator import mul
import math 

import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from einops import rearrange, repeat
from typing import Optional, List

from timm.models.layers import DropPath, Mlp, trunc_normal_
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

        

class D2_EVA02_TEMPLATE(SimpleFeaturePyramid):
    def __init__(self, cfg, input_shape, BACKBONE_TYPE=None ):

        if BACKBONE_TYPE == "lrtk0":
            BACKBONE = EVA02_ViT_LRTKO
        elif BACKBONE_TYPE == 'default':
            BACKBONE = EVA02_ViT
        elif BACKBONE_TYPE == 'template':
            BACKBONE = EVA02_ViT_TEMPLATE
        elif BACKBONE_TYPE == 'lora':
            BACKBONE = EVA02_ViT_LORA
        elif BACKBONE_TYPE == 'dump':
            BACKBONE = EVA02_ViT_DUMP
        
        super().__init__(
            net = BACKBONE(
                img_size= cfg.MODEL.EVA02.IMAGE_SIZE,
                patch_size=cfg.MODEL.EVA02.PATCH_SIZE,
                window_size= cfg.MODEL.EVA02.WINDOW_SIZE,
                embed_dim= cfg.MODEL.EVA02.DMBED_DIM,
                depth= cfg.MODEL.EVA02.DEPTH,
                num_heads= cfg.MODEL.EVA02.NUM_HEADS ,
                drop_path_rate= cfg.MODEL.EVA02.DROP_PATH_RATE,
                mlp_ratio= cfg.MODEL.EVA02.MLP_RATIO,
                # qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes= cfg.MODEL.EVA02.WINDOW_BLOCK_INDEXES,
                # residual_block_indexes=[],
                # use_rel_pos=False,
                use_act_checkpoint = cfg.MODEL.EVA02.CHECKPOINT,
                out_feature="last_feat",
                # intp_freq=True,
            ),
            in_feature = "last_feat",
            out_channels=256,
            scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=cfg.MODEL.EVA02.IMAGE_SIZE,
        )

        pretrained_weight = cfg.MODEL.EVA02.PRETRAINED_WEIGHT 
        if pretrained_weight:
            checkpoint = torch.load(pretrained_weight, map_location='cpu')
            print(f'\nload pretrain weight from {pretrained_weight} \n') 
            self.load_state_dict(checkpoint['model'], strict=False)
        
        self.switch_off = False  
        self.frozen_stages  = -1

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

class EVA02_ViT_TEMPLATE(EVA02_ViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for p in self.parameters():
            p.requires_grad = False
        

        # depth = kwargs['depth']
        # self.frozen_stages = depth + 1
        # self._freeze_stages()
        self.switch_off = False  
        
    # def train(self, mode=True):
    #     for module in self.children():
    #         module.train(mode)

    def swith_off_added_modules(self, ):
        self.switch_off = True 
    
    def swith_on_added_modules(self, ):
        self.switch_off = False 
        
    def _print_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print (" === " , name, param.data.shape)
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 :
            self.pos_embed.requires_grad = False
            for param in self.rope_win.parameters():
                param.requires_grad = False
            for param in self.rope_glb.parameters():
                param.requires_grad = False
            
        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                m = self.blocks[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


@BACKBONE_REGISTRY.register()
class D2_EVA02_VANILLA_TEMPLATE(D2_EVA02_TEMPLATE):
    def __init__(self, cfg, input_shape): 
        super().__init__(cfg, input_shape, BACKBONE_TYPE='template' )

@BACKBONE_REGISTRY.register()
class D2_EVA02_DEFAULT(D2_EVA02_TEMPLATE):
    def __init__(self, cfg, input_shape): 
        super().__init__(cfg, input_shape, BACKBONE_TYPE='default' )


class Attention_FIX(Attention):
    def __init__( self, dim, num_heads=8, attn_head_dim=None,  qkv_bias=True, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, attn_head_dim=attn_head_dim, qkv_bias=qkv_bias, **kwargs)
        
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        
        if qkv_bias:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=True)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=True)
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False )
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False )
        
        del self.v_bias, self.q_bias
        
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 

        ## rope
        q = self.rope(q).type_as(v)
        k = self.rope(k).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1).type_as(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = x.view(B, H, W, C)

        return x

        

            



        

            

 
    
    




##############################################################################
############# LR Tokens    

class EVA02_ViT_LRTKO(EVA02_ViT_TEMPLATE):
    def __init__( self, depth=-1,  drop_path_rate=-1, patch_size=16, embed_dim=-1, 
        prompt_DROPOUT=0.0, prompt_NUM_TOKENS=32, no_norm=True, 
        **kwargs):
        super().__init__(depth=depth, drop_path_rate=drop_path_rate, patch_size=patch_size, embed_dim=embed_dim, **kwargs)

        patch_size = patch_size, patch_size
        self.prompt_dropout = DropPath(prompt_DROPOUT)

        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, embed_dim, prompt_NUM_TOKENS, prompt_NUM_TOKENS))
        nn.init.trunc_normal_(self.prompt_embeddings.data, -val, val)

        # build layers
        prompts = []
        for i_layer in range(depth):
            dim=int(embed_dim)
            deep_prompt_embeddings = nn.Parameter(torch.zeros( 1, dim, prompt_NUM_TOKENS, prompt_NUM_TOKENS))
            nn.init.uniform_(deep_prompt_embeddings.data, -val, val)
            prompts.append(deep_prompt_embeddings)

        self.deep_pompts = nn.ParameterList(prompts)
        # self._print_trainable_params()

    def forward(self, x):
        x = self.patch_embed(x)
        B, H, W = x.shape[:-1]


        if (not self.switch_off):
            prompt_embd = self.prompt_embeddings.expand(B, -1, -1, -1)
            prompt_embd = F.interpolate(prompt_embd, size=(H, W), mode='bicubic', align_corners=True)
            prompt_embd = self.prompt_dropout(prompt_embd)
            prompt_embd = rearrange(prompt_embd, "B C H W -> B H W C")
            x += prompt_embd


        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for i,blk in enumerate(self.blocks):
            if (not self.switch_off):
                x = self.add_tokens(x, H, W, self.deep_pompts[i])
            x = blk(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

    def add_tokens(self, x, H, W, prompt_embd):
        prompt_embd = F.interpolate(prompt_embd, size=(H, W), mode='bicubic', align_corners=True)
        prompt_embd = self.prompt_dropout(prompt_embd)
        prompt_embd = rearrange(prompt_embd, "B C H W -> B H W C")
        x += prompt_embd

        return x

@BACKBONE_REGISTRY.register()
class D2_EVA02_LTTKO(D2_EVA02_TEMPLATE):
    def __init__(self, cfg, input_shape): 
        super().__init__(cfg, input_shape, BACKBONE_TYPE='lrtk0' )






##############################################################################
############# LoRA

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
            if self.bias is not None:
                self.bias.requires_grad = False 
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)  # (3d,) -> (3,d)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)  # (3d,)
        self.reset_parameters()
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

class Block_Lora(Block):
    def __init__(self, rank=1, dim=1024, num_heads=16, qkv_bias=True, rope=None, xattn=True, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope, xattn=xattn, **kwargs)
        
        self.attn = Attention_FIX(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            rope=rope,
            xattn=xattn,
        )

        for p in self.parameters():
            p.requires_grad = False
        
        head_dim = dim // num_heads
        all_head_dim = head_dim * num_heads
        
        self.attn.proj = Linear(dim, dim, r=rank,)
        self.attn.q_proj = MergedLinear(dim, all_head_dim, r=rank, enable_lora=[True], bias=qkv_bias)
        self.attn.k_proj = MergedLinear(dim, all_head_dim, r=rank, enable_lora=[True], bias=False)
        self.attn.v_proj = MergedLinear(dim, all_head_dim, r=rank, enable_lora=[True], bias=qkv_bias)

class EVA02_ViT_LORA(EVA02_ViT_TEMPLATE):
    def __init__( self, depth=12, drop_path_rate=0.0, patch_size=16, embed_dim=768, num_heads=12, 
        mlp_ratio=4*2/3, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=0,
        window_block_indexes=(), residual_block_indexes=(), xattn=True, use_act_checkpoint=False, **kwargs):
        super().__init__(
            depth=depth, drop_path_rate=drop_path_rate, patch_size=patch_size, embed_dim=embed_dim, 
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, window_size=window_size, 
            window_block_indexes=window_block_indexes, residual_block_indexes=residual_block_indexes, xattn=xattn, 
            use_act_checkpoint=use_act_checkpoint, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_Lora(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                rope=self.rope_win if i in window_block_indexes else self.rope_glb,
                xattn=xattn
            )
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.apply(self._init_weights)



@BACKBONE_REGISTRY.register()
class D2_EVA02_LORA(D2_EVA02_TEMPLATE):
    def __init__(self, cfg, input_shape): 
        super().__init__(cfg, input_shape, BACKBONE_TYPE='lora' )







##############################################################################
############# DUMP


class EVA02_ViT_DUMP(EVA02_ViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        structure =[2, 2, 18, 2]
        self.cummulative_index = []
        existing_sum =0 
        for e in structure:
            existing_sum += e 
            self.cummulative_index.append(existing_sum - 1)


    def forward(self, x):
        x = self.patch_embed(x)
        
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        intermediate_laters = []
        # 2 2 18 2 
        for i,blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.cummulative_index:
                intermediate_laters.append( rearrange(x, "B H W C -> B C (H W)").mean(-1).cpu().numpy() )
        
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs, intermediate_laters



@BACKBONE_REGISTRY.register()
class D2_EVA02_DUMP(D2_EVA02_TEMPLATE):
    def __init__(self, cfg, input_shape): 
        super().__init__(cfg, input_shape, BACKBONE_TYPE='dump' )

    def forward(self, x):
        bottom_up_features, intermediate_laters = self.net(x)
        # len(self.net.blocks), self.net.cummulative_index
        # 24 , [1, 3, 21, 23]
        # [e.shape for e in intermediate_laters]
        # [(1, 1024), (1, 1024), (1, 1024), (1, 1024)]
        # print(sum(p.numel() for p in self.net.parameters()))
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                # print("======", self._out_features, self.top_block.in_feature, self._out_features.index(self.top_block.in_feature), results[self._out_features.index(self.top_block.in_feature)].shape)
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        
        # [e.shape for e in results]
        # [torch.Size([1, 256, 192, 192]), torch.Size([1, 256, 96, 96]), torch.Size([1, 256, 48, 48]), torch.Size([1, 256, 24, 24])]
        return {f: res for f, res in zip(self._out_features, results)}, intermediate_laters

