import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import os
import os.path as osp

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import time

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmengine.utils.dl_utils.setup_env import set_multi_processing

from mmdet.utils import setup_cache_size_limit_of_dynamo
from proposed_modules.runner import Runner_Modified
from test import extern_dataset

def fix_path (args, cfg):
    train_root = args.root 
    test_root = args.root 
    if args.root2:
        test_root = args.root2
    
    if "data_root" in cfg:
        if "flickr30k_entities" in cfg.data_root:
            cfg.data_root = os.path.join(train_root , 'flickr_dataset_30k')
        else:
            cfg.data_root = train_root
    if "datasets" in cfg.test_dataloader.dataset:
        for e in cfg.test_dataloader.dataset.datasets:
            if "flickr30k_entities" in cfg.data_root:
                e.data_root = os.path.join(test_root , 'flickr_dataset_30k')
            else:
                e.data_root = e.data_root.replace("data/", test_root)
    
    if "datasets" in cfg.train_dataloader.dataset:
        for e in cfg.train_dataloader.dataset.datasets:
            if ("data_root" in e and "flickr30k_entities" in e.data_root) :
                e.data_root = os.path.join(train_root , 'flickr_dataset_30k/flickr30k/')
            elif ("data_root" in e.dataset and "flickr30k_entities" in e.dataset.data_root):
                e.dataset.data_root = os.path.join(train_root , 'flickr_dataset_30k/flickr30k/')
            else:
                assert False, "training on other datasets not yet verified "
        
    if "metrics" in cfg.val_evaluator:
        for e in cfg.val_evaluator.metrics:
            if "ann_file" in e:
                e.ann_file = e.ann_file.replace("data/", test_root)
    for e in cfg:
        if "val_evaluator_" in e:
            if "ann_file" in cfg[e]:
                cfg[e].ann_file = cfg[e].ann_file.replace("data/", test_root)
        if "dataset_" in e and "data_root" in cfg[e]:
            if "ann_file" in cfg[e]:
                cfg[e].data_root = cfg[e].data_root.replace("data/", test_root)

    # for e in cfg.datasets:
    #     e.data_root = e.data_root.replace("data/", train_root)
    if "metrics" in cfg:
        for e in cfg.metrics:
            if "ann_file" in e:
                e.ann_file = e.ann_file.replace("data/", test_root)
    if "metrics" in cfg.test_evaluator:
        for e in cfg.test_evaluator.metrics:
            if "ann_file" in e:
                e.ann_file = e.ann_file.replace("data/", test_root)
    if "datasets" in cfg.val_dataloader.dataset:
        for e in cfg.val_dataloader.dataset.datasets:
            e.ann_file = e.ann_file.replace("data/", test_root)
    
    ##### LVIS 
    if "data_root" in cfg.val_dataloader.dataset:
        cfg.val_dataloader.dataset.data_root = cfg.val_dataloader.dataset.data_root.replace("data/", test_root)
        cfg.test_dataloader.dataset.data_root = cfg.test_dataloader.dataset.data_root.replace("data/", test_root)
        if args.diff_dataset == "lvis":
            cfg.val_dataloader.dataset.data_root = cfg.val_dataloader.dataset.data_root.replace("coco", "lvis") 
            cfg.test_dataloader.dataset.data_root = cfg.test_dataloader.dataset.data_root.replace("coco", "lvis") 

    if "ann_file" in cfg.val_evaluator and "data/" in cfg.val_evaluator.ann_file:
        cfg.val_evaluator.ann_file = cfg.val_evaluator.ann_file.replace("data/", test_root)
        cfg.test_evaluator.ann_file = cfg.test_evaluator.ann_file.replace("data/", test_root)
        if args.diff_dataset == "lvis":
            cfg.val_evaluator.ann_file = cfg.val_evaluator.ann_file.replace("coco", "lvis")
            cfg.test_evaluator.ann_file = cfg.test_evaluator.ann_file.replace("coco", "lvis")
        
    ##### ODWIN-35 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if "metrics" in cfg.val_evaluator:
        for e in cfg.val_evaluator.metrics:
            if "ann_file" in e and "Odinw35_new_annotations" in e.ann_file:
                e.ann_file = os.path.join(parent_dir, e.ann_file,)
    for e in cfg:
        if "val_evaluator_" in e:
            if "ann_file" in cfg[e] and "Odinw35_new_annotations" in cfg[e].ann_file:
                cfg[e].ann_file = os.path.join(parent_dir, cfg[e].ann_file)
    if "metrics" in cfg:
        for e in cfg.metrics:
            if "ann_file" in e and "Odinw35_new_annotations" in e.ann_file:
                e.ann_file = os.path.join(parent_dir, e.ann_file)
    if "metrics" in cfg.test_evaluator:
        for e in cfg.test_evaluator.metrics:
            if "ann_file" in e and "Odinw35_new_annotations" in e.ann_file:
                e.ann_file = os.path.join(parent_dir, e.ann_file)
    if "datasets" in cfg.val_dataloader.dataset:
        for e in cfg.val_dataloader.dataset.datasets:
            if "ann_file" in e and "Odinw35_new_annotations" in e.ann_file:
                e.ann_file = os.path.join(parent_dir, e.ann_file)
    if "datasets" in cfg.test_dataloader.dataset:
        for e in cfg.test_dataloader.dataset.datasets:
            if "ann_file" in e and "Odinw35_new_annotations" in e.ann_file:
                e.ann_file = os.path.join(parent_dir, e.ann_file)

    
    
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    
    parser.add_argument('--no-display', action='store_true', help='no logging')
    parser.add_argument('--runner', type=str, default=None, help='')
    parser.add_argument('--root', type=str, help='')
    parser.add_argument('--root2', type=str, help='', default=None)
    parser.add_argument('--subset', type=str, default=None, help='')
    parser.add_argument('--diff-dataset', type=str, default=None, help='')
    parser.add_argument('--detector', type=str, default=None, help='')

    parser.add_argument('--num_workers', type=int, default=None, help='')
    parser.add_argument('--batch_size', type=int, default=None, help='')

    parser.add_argument('--sanity-check', action='store_true', help='no logging')
    parser.add_argument('--severity', type=int, default=None, help='no logging')

    parser.add_argument('--train-mode', type=str, default=None, help='')


    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = osp.join('./work_dirs', args.work_dir)
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.root:
        fix_path (args, cfg)

    
    # if args.train_mode in ["motion_blur2", "rain_model2", "snow_model2", "atmospheric"]:
    #     print("SPAWN....")
    #     cfg.env_cfg.mp_cfg.mp_start_method = "spawn"
    #     cfg.env_cfg.mp_cfg.opencv_num_threads = 4
    #     if args.train_mode in ["rain_model2", "snow_model2", "atmospheric"]:
    #         cfg.env_cfg.mp_cfg.opencv_num_threads = 4
    #         # cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // 2
    #         cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size * 2
    #         cfg.train_dataloader.num_workers = 8 
    if args.train_mode in ["motion_blur2", "rain_model3", "snow_model3", "atmospheric2", "snow_model2", "rain_model2"]:
        print("SPAWN....")
        cfg.env_cfg.mp_cfg.mp_start_method = "spawn"
        cfg.env_cfg.mp_cfg.opencv_num_threads = 4
    
        
        
    if args.num_workers:
        cfg.train_dataloader.num_workers = args.num_workers
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size

    if args.sanity_check:
        cfg.train_cfg.sanity_check = True 

    if args.severity is not None:
        import pdb
        pdb.set_trace()
        args.severity
        cfg.train_cfg.sanity_check = True 
        
    cfg.load_from = args.checkpoint
    if args.detector :
        from proposed_modules import model
        cfg.model.type = args.detector


    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    if args.train_mode:
        for dataset in cfg.train_dataloader.dataset.datasets:
            if ("data_root" in dataset and "flickr_dataset_30k" in dataset.data_root):
                dataset.mode = args.train_mode
            elif ("data_root" in dataset.dataset and "flickr_dataset_30k" in dataset.dataset.data_root):
                dataset.dataset.mode = args.train_mode
            if args.subset:
                dataset.ann_file  = args.subset
            if args.train_mode in ["rain_model3", "snow_model3", "atmospheric2"]:
                dataset.data_root = dataset.data_root.replace("flickr30k", "Perturbed")
                if args.train_mode == "rain_model3":
                    dataset.data_root = os.path.join(dataset.data_root, "RAIN")
                if args.train_mode == "snow_model3":
                    dataset.data_root = os.path.join(dataset.data_root, "SNOW")
                if args.train_mode == "atmospheric2":
                    dataset.data_root = os.path.join(dataset.data_root, "TURBULENCE")
                
    
    if args.diff_dataset:
        dataset_dict = extern_dataset()
        cfg.test_evaluator['type'] = 'Ext_evaluate'

        if args.diff_dataset == 'WEDGE':
            cfg.test_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'WEDGE/')
            cfg.test_dataloader.dataset.ann_file = dataset_dict['wedge_test']['ann_file']
            cfg.test_dataloader.dataset['data_prefix']['img'] = dataset_dict['wedge_test']['img_dir']
            cfg.test_evaluator['ann_file'] = os.path.join(cfg.data_root, 'WEDGE', dataset_dict['wedge_test']['ann_file'])
            
            
            cfg.val_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'WEDGE/')
            cfg.val_dataloader.dataset.ann_file = dataset_dict['wedge_test']['ann_file']
            cfg.val_dataloader.dataset['data_prefix']['img'] = dataset_dict['wedge_test']['img_dir']
            cfg.val_evaluator['ann_file'] = os.path.join(cfg.data_root, 'WEDGE', dataset_dict['wedge_test']['ann_file'])
            
            
            
            cfg.train_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'WEDGE/')
            cfg.train_dataloader.dataset.ann_file = dataset_dict['wedge_test']['ann_file'].replace('test', 'train')
            cfg.train_dataloader.dataset['data_prefix']['img'] = dataset_dict['wedge_test']['img_dir']
            cfg.train_dataloader.dataset.type = 'Extern_Dataset_CoCo_train'
            
            # # for e in cfg.train_dataloader.dataset.datasets:e
            # cfg.train_dataloader.dataset.datasets[0]  = cfg.coco_dataset            
            # cfg.train_dataloader.dataset.datasets[0].data_root = os.path.join(cfg.data_root, 'WEDGE/')
            # cfg.train_dataloader.dataset.datasets[0].ann_file = dataset_dict['wedge_test']['ann_file'].replace('test', 'train')
            # cfg.train_dataloader.dataset.datasets[0]['data_prefix']['img'] = dataset_dict['wedge_test']['img_dir']
            
             
            
            
            
    # resume is determined in this priority: resume from > auto_resume
    # if args.resume == 'auto':
    #     cfg.resume = True
    #     cfg.load_from = None
    # elif args.resume is not None:
    #     cfg.resume = True
    #     cfg.load_from = args.resume


    # build the runner from config
    if args.runner: 
        if args.runner == 'Runner_Modified':
            cfg.no_display = args.no_display
            runner = Runner_Modified.from_cfg(cfg)
    elif 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    start_time = time.time()
    # start training
    runner.train()

    end_time = time.time()
    
    elapsed = int(end_time - start_time)
    days = elapsed // 86400
    hours = (elapsed % 86400) // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    print(" \n\n TRAINING COMPLETE .... \n\n ")
    print(f"Elapsed time: {days}d:{hours}h:{minutes}m:{seconds}s")



if __name__ == '__main__':
    main()