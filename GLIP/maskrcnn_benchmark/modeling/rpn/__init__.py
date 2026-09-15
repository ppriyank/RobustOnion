# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from .rpn import build_rpn
from .rpn import RPNModule
from .retina import RetinaNetModule
from .fcos import FCOSModule
from .atss import ATSSModule
from .dyhead import DyHeadModule
from .vldyhead import VLDyHeadModule

from .vldyhead_proposed import VLDyHeadModule_LANG_TRI, VLDyHeadModule_LANG_Distill, VLDyHeadModule_Dump, \
    VLDyHeadModule_RobustSAM, VLDyHeadModule_LANG_FUSE

_RPN_META_ARCHITECTURES = {"RPN": RPNModule,
                           "RETINA": RetinaNetModule,
                           "FCOS": FCOSModule,
                           "ATSS": ATSSModule,
                           "DYHEAD": DyHeadModule,
                           "VLDYHEAD": VLDyHeadModule,

                           'VLDyHeadModule_LANG_TRI' : VLDyHeadModule_LANG_TRI, 
                           'VLDyHeadModule_LANG_Distill': VLDyHeadModule_LANG_Distill, 

                           'VLDyHeadModule_Dump': VLDyHeadModule_Dump, 
                           
                           'VLDyHeadModule_RobustSAM': VLDyHeadModule_RobustSAM, 
                           'VLDyHeadModule_LANG_FUSE': VLDyHeadModule_LANG_FUSE, 
                           }


def build_rpn(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    rpn_arch = _RPN_META_ARCHITECTURES[cfg.MODEL.RPN_ARCHITECTURE]
    return rpn_arch(cfg)
