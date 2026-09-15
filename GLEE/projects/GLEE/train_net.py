#!/usr/bin/env python3
# Copyright (c) 2024 ByteDance. All Rights Reserved.
# GLEE Training Script.
# GLEE: General Object Foundation Model for Images and Videos at Scale (CVPR 2024)
# https://arxiv.org/abs/2312.09158
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import sys 
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(current_dir))
sys.path = [root] + sys.path
# sys.path.append(root)


import itertools
import time
from typing import Any, Dict, List, Set
import yaml
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators, LVISEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping


from detectron2.projects.glee import add_glee_config, build_detection_train_loader, build_detection_test_loader
from detectron2.projects.glee.backbone.eva01 import get_vit_lr_decay_rate
from detectron2.projects.glee.data import (
    get_detection_dataset_dicts, RefCOCODatasetMapper, YTVISDatasetMapper,  OMNILABEL_Evaluator, Joint_Image_LSJDatasetMapper,Joint_Image_Video_LSJDatasetMapper,  COCO_CLIP_DatasetMapper,UnivideoimageDatasetMapper , UnivideopseudoDatasetMapper, 
    RefCOCODatasetMapper_Perturb, Joint_Image_LSJDatasetMapper_Perturb, Joint_Image_Video_LSJDatasetMapper_Perturb, External_Dataset
)
from detectron2.projects.glee.data import build_custom_train_loader,YTVISEvaluator
# from mmengine.utils.dl_utils.setup_env import set_multi_processing

from detectron2.projects.glee.engine import DefaultTrainer_Custom, External_eval



# class Trainer(DefaultTrainer):
class Trainer(DefaultTrainer_Custom):
    """
    Extension of the Trainer class adapted.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "lvis":
            evaluator_list.append(LVISEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "coco":
            if "objects365" in dataset_name or "openimage" in dataset_name or "bdd_det" in dataset_name  or "odinw" in dataset_name or "flicker" in dataset_name:
                force_tasks = {"bbox"}
            elif cfg.FORCE_TASK_BOX:
                force_tasks = {"bbox"}
            else:
                force_tasks = None
            if "refcoco" in dataset_name:
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder, force_tasks=force_tasks, refcoco=True))
            elif dataset_name in ("dawn-test", "bddk-test", "wedge-test", "FoggyCitiscape-a-modal-test", "FoggyCitiscape-modal-test"):
                force_tasks = {"bbox"}
                evaluator_list.append(External_eval(dataset_name=dataset_name, tasks=cfg, distributed=True, output_dir=output_folder, force_tasks=force_tasks, refcoco=False))
            else:
                evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder, force_tasks=force_tasks, refcoco=False))
        elif evaluator_type == "ytvis":
            evaluator_list.append(YTVISEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "omnilabel":
            evaluator_list.append(OMNILABEL_Evaluator(dataset_name, cfg, True, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        _root = os.getenv("DETECTRON2_DATASETS", "datasets")
        if cfg.DATALOADER.SAMPLER_TRAIN == "MultiDatasetSampler":
            # multiple datasets (for example, detection & grounding)
            datasetname = ' '.join(list(cfg.DATASETS.TRAIN))
            if cfg.MODEL.PSEUDO_VIDEO:   
                mapper = UnivideopseudoDatasetMapper(cfg, is_train=True)   
            else:  
                if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj':
                    mapper = Joint_Image_Video_LSJDatasetMapper_Perturb(cfg, is_train=True, root=_root)
                else:
                    mapper = UnivideoimageDatasetMapper(cfg, is_train=True)   
            data_loader = build_custom_train_loader(cfg, mapper=mapper)   
            return data_loader
        else:
            assert False, "not yet verified ... "
        
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if  'ytvis' in dataset_name or 'ovis' in dataset_name or 'video' in dataset_name or 'rvos' in dataset_name or 'ytbvos' in dataset_name or 'lvvis' in dataset_name:
            if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj':
                mapper = Joint_Image_Video_LSJDatasetMapper(cfg, is_train=False)
            else:
                mapper = YTVISDatasetMapper(cfg, is_train=False)
        else:
            if cfg.INPUT.DATASET_MAPPER_NAME == 'coco_instance_lsj':
                if cfg.DATASETS.DATALOADER_MODE:
                    mapper = Joint_Image_LSJDatasetMapper_Perturb(cfg, is_train=False)
                else:
                    mapper = Joint_Image_LSJDatasetMapper(cfg, is_train=False)
            elif cfg.DATASETS.DATALOADER_MODE and (not dataset_name.startswith('lvis')):
                mapper = RefCOCODatasetMapper_Perturb(cfg, is_train=False)
            elif cfg.INPUT.DATASET_MAPPER_NAME == 'ext_dataset':
                # external_name
                mapper = External_Dataset(cfg, is_train=False)
            else:
                if dataset_name.startswith('lvis'):
                    return build_detection_test_loader(cfg, dataset_name)
                mapper = RefCOCODatasetMapper(cfg, is_train=False)
        loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return loader

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                if cfg.SOLVER.LR_DECAY_RATE is not None:
                    backbone_decay = get_vit_lr_decay_rate(key, cfg.SOLVER.LR_DECAY_RATE, cfg.SOLVER.LR_DECAY_RATE_NUM_LAYERS)
                    lr = lr * backbone_decay
                    # print(key, ' lr decay=',backbone_decay)
                    
            if "text_encoder" in key:
                lr = lr * cfg.SOLVER.TEXTENCODER_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer




def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        # if ori_value:
        #     logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")


def load_opt_from_config_files(conf_file):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files: config file path

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    load_config_dict_to_opt(opt, config_dict)

    return opt

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_glee_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.grad_cam:
        cfg.MODEL.GRADCAM = True 
    cfg.freeze()  
    default_setup(cfg, args)
    if args.spawn_method:
        # if cfg.DATASETS.DATALOADER_MODE in ["motion_blur2", "rain_model2", "snow_model2", "atmospheric"]:
        print("SPAWN....")
        # set_multi_processing(mp_start_method='spawn', opencv_num_threads=4, distributed=True)
        torch.multiprocessing.set_start_method('spawn')
    return cfg


def main(args):
    cfg = setup(args)
    if args.grad_cam:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        Trainer.grad_cam(cfg, model)
        return 
    if args.dump_feats:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        Trainer.test_DUMP(cfg=cfg, model=model, start_index=args.dump_start, end_index=args.dump_end)
        return 
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg=cfg, model=model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    if_resume = args.resume
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=if_resume)
    return trainer.train()


if __name__ == "__main__":
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
