
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY

from .swin import Mlp, window_partition, window_reverse, WindowAttention, SwinTransformerBlock, \
    PatchMerging, BasicLayer, PatchEmbed, SwinTransformer, D2SwinTransformer


from einops import rearrange, repeat



@BACKBONE_REGISTRY.register()
class D2SwinTransformer_Dump(D2SwinTransformer):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        
    def forward(self, x):
        assert (x.dim() == 4), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]

        # for e in outputs:outputs[e].shape
        intermediate_feats = [rearrange(outputs[e], "B C H W -> B C (H W)").mean(-1).cpu()  for e in outputs]        
        

        return outputs, intermediate_feats




