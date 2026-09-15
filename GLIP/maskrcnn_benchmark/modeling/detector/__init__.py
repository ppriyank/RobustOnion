from .generalized_rcnn import GeneralizedRCNN
from .generalized_vl_rcnn import GeneralizedVLRCNN

from .proposed_detector import GeneralizedVLRCNN_PROPOSE, GeneralizedVLRCNN_TRILANG, GeneralizedVLRCNN_TRILANG_DEBUG, GeneralizedVLRCNN_TRILANG_DUMP, GeneralizedVLRCNN_TRILANG_GFLOP, \
    GeneralizedVLRCNN_DISTILL, GeneralizedVLRCNN_DUMPING, Feat_Comp, GeneralizedVLRCNN_VAR, GeneralizedVLRCNN_RobustSAM, GeneralizedVLRCNN_RL

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedVLRCNN": GeneralizedVLRCNN,

                                 'GeneralizedVLRCNN_PROPOSE': GeneralizedVLRCNN_PROPOSE, 
                                 'GeneralizedVLRCNN_TRILANG': GeneralizedVLRCNN_TRILANG, 
                                 'GeneralizedVLRCNN_TRILANG_DEBUG': GeneralizedVLRCNN_TRILANG_DEBUG, 
                                 'GeneralizedVLRCNN_TRILANG_DUMP': GeneralizedVLRCNN_TRILANG_DUMP, 
                                 'GeneralizedVLRCNN_TRILANG_GFLOP': GeneralizedVLRCNN_TRILANG_GFLOP, 
                                 
                                 'GeneralizedVLRCNN_DISTILL': GeneralizedVLRCNN_DISTILL, 
                                 'GeneralizedVLRCNN_DUMPING': GeneralizedVLRCNN_DUMPING, 

                                 'GeneralizedVLRCNN_VAR': GeneralizedVLRCNN_VAR, 
                                 'GeneralizedVLRCNN_RobustSAM': GeneralizedVLRCNN_RobustSAM,
                                 'GeneralizedVLRCNN_RL': GeneralizedVLRCNN_RL, 

                                 'Feat_Comp': Feat_Comp, 
                                 }


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
