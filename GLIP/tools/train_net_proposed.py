import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
 
import torch
from maskrcnn_benchmark.config import cfg, try_to_find
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine import trainer as TRAINER_FNS
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, reduce_sum, broadcast_data
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import (MetricLogger, TensorboardLogger)
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import numpy as np
import random
from maskrcnn_benchmark.utils.amp import autocast, GradScaler

from train_net import setup_for_distributed
from mmengine.utils.dl_utils.setup_env import set_multi_processing

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

def compute_flops(model, verbose=False, print_per_layer_stat=False):

    resolution = (1, 3, 600, 600)
    input = torch.randn(resolution)
    outs = model.backbone.body(input.cuda())
    print([e.shape for e in outs])

    model.backbone.body.GFLOP = True 
    # python -m pip install thop fvcore
    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model.backbone.body.float(), input.cuda())
    print(f" \n\n***** FLOP TOTAL : {flops.total() / 10 ** 9}" )
    per_modules = flops.by_module()
    # keyss = {key for key in per_modules if "block" in key and "attn" in key and per_modules[key] != 0 and "norm" not in key and 'proj' not in key}
    # selected_per_module = {key: per_modules[key] for key in keyss}
    # print(f"FLOP BY MODULES : {selected_per_module}" )
    # print(f"FLOP BY MODULES & OPERATOR : {flops.by_module_and_operator()}" )

    # from maskrcnn_benchmark.utils.stats import get_model_complexity_info
    # params, flops = get_model_complexity_info(model.backbone.body.float(), resolution, input_constructor=lambda x: torch.rand(x).cuda())
        
        

    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print(f" Backbone parameters: {np.sum([int(np.prod(p.shape)) for p in model.backbone.body.parameters()]):,}")

    
def vanilla_load(model, distributed, checkpoint):
    if distributed:
        extra_checkpoint_data = model.load_state_dict( checkpoint['model'], strict=False)
    else:
        checkpoint = {k.replace("module.", ""):v for k,v in checkpoint['model'].items()}
        extra_checkpoint_data = model.load_state_dict( checkpoint, strict=False)
    print(extra_checkpoint_data)
    return model 

def setup_model(cfg, device):
    model = build_detection_model(cfg)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if 'running_mean' in name:
                torch.nn.init.constant_(param, 0)
            if 'running_var' in name:
                torch.nn.init.constant_(param, 1)
    
    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    if cfg.MODEL.BACKBONE.FREEZE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        print("LANGUAGE_BACKBONE FROZEN.")
        for p in model.language_backbone.body.parameters():
            p.requires_grad = False

    if cfg.MODEL.FPN.FREEZE:
        for p in model.backbone.fpn.parameters():
            p.requires_grad = False
    if cfg.MODEL.RPN.FREEZE:
        for p in model.rpn.parameters():
            p.requires_grad = False
    
    return model

def load_weights(cfg, model, optimizer, scheduler, arguments, data_loader, distributed):
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    # Loading only 
    # checkpoint = torch.load(try_to_find(cfg.MODEL.WEIGHT))
    # print(extra_checkpoint_data)
    if cfg.MODEL.BACKBONE_TYPE and 'FAN' in cfg.MODEL.BACKBONE_TYPE:
        checkpoint = torch.load(try_to_find(cfg.MODEL.WEIGHT))
        keys = list(checkpoint['model'].keys())
        for key in keys:
            if 'backbone.body.layers' in key and 'mlp.fc' in key: 
                old_key = key    
                new_key = key.replace('mlp', 'mlp.mlp')
                checkpoint['model'][new_key] = checkpoint['model'][old_key]
                del checkpoint['model'][old_key]
                # print("--", key)
        # backbone.body.layers.0.blocks.1.mlp.mlp.fc2.weight torch.Size([96, 384])
        model = vanilla_load(model, distributed, checkpoint)
    else:
        extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT), force=True, skip_optimizer=True)
        arguments.update(extra_checkpoint_data)
    # model.language_backbone.body.model.embeddings.position_id

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    if cfg.SOLVER.MAX_EPOCH  != 0 :
        checkpoint_period = len(data_loader) // cfg.SOLVER.MAX_EPOCH
    cfg.defrost()
    cfg.SOLVER.CHECKPOINT_PERIOD= checkpoint_period
    cfg.freeze()

    return checkpointer, checkpoint_period
    
def vanilla_testing_fn(model, args, distributed, data_loaders_val, cfg, display_results=True, return_metric='AP50'):
    dataset_name="coco"
    model.eval()
    if not (args.no_switchoff):
        print("\n\n .. Switching off modules" )
        if distributed:
            model.module.switch_off_modules()
        else:
            model.switch_off_modules()
    results = inference(
            model,
            data_loaders_val,
            dataset_name=dataset_name, iou_types=("bbox",),
            box_only=cfg.MODEL.RPN_ONLY and (cfg.MODEL.RPN_ARCHITECTURE == "RPN" or cfg.DATASETS.CLASS_AGNOSTIC), 
            device=cfg.MODEL.DEVICE, expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL, output_folder=None, cfg=cfg,
            display_results= display_results, 
        )
    synchronize()
    if not (args.no_switchoff):
        print("\n\n .. Switching on modules" )
        if distributed:
            model.module.switch_on_modules()
        else:
            model.switch_on_modules()
    # model.module.switch_off_modules()

    device = f"cuda:{get_rank()}"
    if results is None :
        results = torch.tensor(0.0, device=device, dtype=torch.float64)
    else:
        results = [results[0].results['bbox'][return_metric]]
        results = torch.tensor(results, device=device, dtype=torch.float64)
    
    results = reduce_sum(results)
    return results


def train(cfg, local_rank, distributed, use_tensorboard=False, args=None):
    
    device = torch.device(cfg.MODEL.DEVICE)
    model = setup_model(cfg, device)
    
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=0  # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    )

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        data_loaders_val = data_loaders_val[0]
    else:
        data_loaders_val = None

    
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )

    arguments = {}
    arguments["iteration"] = 0
    checkpointer, checkpoint_period = load_weights(cfg, model, optimizer, scheduler, arguments, data_loader, distributed)
    
    
    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=cfg.OUTPUT_DIR,
            start_iter=arguments["iteration"],
            delimiter="  "
        )
    else:
        meters = MetricLogger(delimiter="  ")
    
    if args.GFLOP:
        compute_flops(model, verbose=False, print_per_layer_stat=False)
        quit()
        
    if args.vanilla_testing:
        vanilla_testing_fn(model, args, distributed, data_loaders_val, cfg)
    
    print('Trainiable parameters: ')
    print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters() if p.requires_grad == True]):,}")
    # quit()
    
    extra_args = {}
    model.train()
    TRAIN_FN = do_train
    if args.train_fn:
        print(f"***** {args.train_fn} **** ")
        TRAIN_FN = getattr(TRAINER_FNS, args.train_fn)
        extra_args = dict(
            display_images = args.display_images, 

        )
    TRAIN_FN(
        cfg=cfg, model=model, data_loader=data_loader, optimizer=optimizer, scheduler=scheduler,
        checkpointer=checkpointer, device=device, checkpoint_period=checkpoint_period,
        arguments=arguments, val_data_loader=data_loaders_val, meters=meters, 
        no_display = args.no_display, **extra_args)
    return model



def args_parse(parse_params=True):
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument("--no-display", default=False, action='store_true',)
    parser.add_argument("--train_fn", default=None, type=str,)
    parser.add_argument( "--subset", default=None, type=str)
    parser.add_argument( "--vanilla-testing", default=False, action='store_true',)
    parser.add_argument( "--display_images", default=None, type=str,)
    parser.add_argument( "--spawn-method", default=False, action='store_true',)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--no-switchoff", default=False, action='store_true',)
    parser.add_argument("--GFLOP", default=False, action='store_true',)
    
    parser.add_argument("--debug", default=False, action='store_true',)

    parser.add_argument("--use-tensorboard",
                        dest="use_tensorboard",
                        help="Use tensorboardX logger (Requires tensorboardX installed)",
                        action="store_true",
                        default=False
                        )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument("--save_original_config", action="store_true")
    parser.add_argument("--disable_output_distributed", action="store_true")
    parser.add_argument("--override_output_dir", default=None)

    if parse_params:
        args = parser.parse_args()
    else:
        args = None 
    return args, parser

def basic_setup(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.spawn_method:
        set_multi_processing(mp_start_method='spawn', opencv_num_threads=4, distributed=args.distributed)
        # torch.multiprocessing.set_start_method('spawn')
    if args.distributed:        
        import datetime
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            timeout=datetime.timedelta(0, 7200)
        )
    if args.disable_output_distributed:
        setup_for_distributed(args.local_rank <= 0)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # specify output dir for models
    if args.override_output_dir:
        cfg.OUTPUT_DIR = args.override_output_dir
    
    if args.subset:
        print( f"****** {args.subset} ***** ")
        new_train_set = [e + f"_{args.subset}" for e in cfg.DATASETS.TRAIN]
        new_train_set = tuple(new_train_set)
        cfg.DATASETS.TRAIN = new_train_set
    cfg.freeze()
    
    ### seeding
    seed = cfg.SOLVER.SEED + args.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    ### storage
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    
    ### logger related 
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    if args.no_display:
        import logging
        logger.setLevel(logging.WARNING)
    
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    if args.save_original_config:
        import shutil
        shutil.copy(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config_original.yml'))
    save_config(cfg, output_config_path)
    
    return args, num_gpus, cfg

    
    



def main():
    
    args, parser = args_parse()
    args, num_gpus, cfg = basic_setup(args)
    model = train(cfg=cfg,
                  local_rank=args.local_rank,
                  distributed=args.distributed,
                  use_tensorboard=args.use_tensorboard,
                  args=args)


if __name__ == "__main__":
    main()
