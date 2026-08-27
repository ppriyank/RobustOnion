import os
import os.path as osp

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))




# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from proposed_modules.runner import Runner_Modified, Runner_Dump


from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

import numpy as np 
# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument('--root', type=str, help='')
    parser.add_argument('--flop_analysis', action='store_true')
    parser.add_argument('--diff-dataset', type=str, default=None, help='')
    
    parser.add_argument('--detector', type=str, default=None, help='')
    parser.add_argument('--runner', type=str, default=None, help='')
    parser.add_argument('--sub_dataset', type=str, default=None, help='')
    
    parser.add_argument('--start-index', type=int, default=None, help='')
    parser.add_argument('--end-index', type=int, default=None, help='')
    
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--no-display', action='store_true', help='no logging')
    parser.add_argument('--severity', type=int, default=None, help='severity of noise')

    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def extern_dataset():
    return {
    "bdd100k_val": {
            'img_dir': "images/100k/val", 
            'ann_file': "labels_coco/bdd100k_labels_images_val_coco.json"
    },
    "dawn_test": {
            'img_dir': "./",
            'ann_file': "labels_coco/DAWN_test.json"
    },
    "wedge_test": {
            'img_dir': "./",
            'ann_file': "labels_coco/WEDGE_test.json"
    },
    "FoggyCityscape_amodal_val": {
            'img_dir': "val/",
            'ann_file': "labels_coco/Cityscape_labels_val_amodal.json"
        },

    "FoggyCityscape_modal_val": {
        'img_dir': "val/",
        'ann_file': "labels_coco/Cityscape_labels_val_modal.json"
    },
    }
        


def fix_path (args, cfg):
    if args.root:
        if "data_root" in cfg:
            if "flickr30k_entities" in cfg.data_root:
                cfg.data_root = os.path.join(args.root , 'flickr_dataset_30k')
            else:
                cfg.data_root = args.root
        if "datasets" in cfg.test_dataloader.dataset:
            for e in cfg.test_dataloader.dataset.datasets:
                if "flickr30k_entities" in cfg.data_root:
                    e.data_root = os.path.join(args.root , 'flickr_dataset_30k')
                else:
                    e.data_root = e.data_root.replace("data/", args.root)
        if "metrics" in cfg.val_evaluator:
            for e in cfg.val_evaluator.metrics:
                if "ann_file" in e:
                    e.ann_file = e.ann_file.replace("data/", args.root)
        for e in cfg:
            if "val_evaluator_" in e:
                if "ann_file" in cfg[e]:
                    cfg[e].ann_file = cfg[e].ann_file.replace("data/", args.root)
            if "dataset_" in e and "data_root" in cfg[e]:
                if "ann_file" in cfg[e]:
                    cfg[e].data_root = cfg[e].data_root.replace("data/", args.root)

        # for e in cfg.datasets:
        #     e.data_root = e.data_root.replace("data/", args.root)
        if "metrics" in cfg:
            for e in cfg.metrics:
                if "ann_file" in e:
                    e.ann_file = e.ann_file.replace("data/", args.root)
        if "metrics" in cfg.test_evaluator:
            for e in cfg.test_evaluator.metrics:
                if "ann_file" in e:
                    e.ann_file = e.ann_file.replace("data/", args.root)
        if "datasets" in cfg.val_dataloader.dataset:
            for e in cfg.val_dataloader.dataset.datasets:
                e.ann_file = e.ann_file.replace("data/", args.root)
        
        ##### LVIS 
        if "data_root" in cfg.val_dataloader.dataset:
            cfg.val_dataloader.dataset.data_root = cfg.val_dataloader.dataset.data_root.replace("data/", args.root)
            cfg.test_dataloader.dataset.data_root = cfg.test_dataloader.dataset.data_root.replace("data/", args.root)
            if args.diff_dataset == "lvis":
                cfg.val_dataloader.dataset.data_root = cfg.val_dataloader.dataset.data_root.replace("coco", "lvis") 
                cfg.test_dataloader.dataset.data_root = cfg.test_dataloader.dataset.data_root.replace("coco", "lvis") 

        if "ann_file" in cfg.val_evaluator and "data/" in cfg.val_evaluator.ann_file:
            cfg.val_evaluator.ann_file = cfg.val_evaluator.ann_file.replace("data/", args.root)
            cfg.test_evaluator.ann_file = cfg.test_evaluator.ann_file.replace("data/", args.root)
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





def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    if args.dist_backend:
        cfg.env_cfg.dist_cfg.backend = args.dist_backend
    
    if args.root:
        fix_path (args, cfg)

        
    if args.detector :
        from proposed_modules import model
        cfg.model.type = args.detector

    if args.sub_dataset and cfg.test_dataloader.dataset.type == 'ConcatDataset':
        for e in cfg.test_dataloader.dataset.datasets :
            e.type = args.sub_dataset

    if 'mode' in cfg.test_dataloader.dataset and cfg.test_dataloader.dataset.mode and cfg.test_dataloader.dataset.type == 'ConcatDataset':
        for e in cfg.test_dataloader.dataset.datasets:
            e.type = args.sub_dataset
            e.mode = cfg.test_dataloader.dataset.mode
        del cfg.test_dataloader.dataset.mode
    
    if args.severity is not None:
        if cfg.test_dataloader.dataset.type == 'ConcatDataset':
            for e in cfg.test_dataloader.dataset.datasets:
                e.severity = args.severity
        else:
            import pdb
            pdb.set_trace()
        
    if args.diff_dataset:
        dataset_dict = extern_dataset()
        cfg.test_evaluator['type'] = 'Ext_evaluate'
        
        if args.diff_dataset == 'BDDK':
            cfg.test_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'bdd100k')
            cfg.test_dataloader.dataset.ann_file = dataset_dict['bdd100k_val']['ann_file']
            cfg.test_dataloader.dataset['data_prefix']['img'] = dataset_dict['bdd100k_val']['img_dir']
            cfg.test_evaluator['ann_file'] = os.path.join(cfg.data_root, 'bdd100k', dataset_dict['bdd100k_val']['ann_file'])

        elif args.diff_dataset == 'DAWN':
            cfg.test_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'DAWN2/')
            cfg.test_dataloader.dataset.ann_file = dataset_dict['dawn_test']['ann_file']
            cfg.test_dataloader.dataset['data_prefix']['img'] = dataset_dict['dawn_test']['img_dir']
            cfg.test_evaluator['ann_file'] = os.path.join(cfg.data_root, 'DAWN2', dataset_dict['dawn_test']['ann_file'])
            
        
        elif args.diff_dataset == 'WEDGE':
            cfg.test_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'WEDGE/')
            cfg.test_dataloader.dataset.ann_file = dataset_dict['wedge_test']['ann_file']
            cfg.test_dataloader.dataset['data_prefix']['img'] = dataset_dict['wedge_test']['img_dir']
            cfg.test_evaluator['ann_file'] = os.path.join(cfg.data_root, 'WEDGE', dataset_dict['wedge_test']['ann_file'])
            
        elif 'FoggyCity' in args.diff_dataset:
            cfg.test_dataloader.dataset.data_root = os.path.join(cfg.data_root, 'leftImg8bit_foggyDBF/')
            cfg.test_dataloader.dataset['data_prefix']['img'] = dataset_dict['FoggyCityscape_amodal_val']['img_dir']
            if args.diff_dataset == 'FoggyCity_A':
                cfg.test_dataloader.dataset.ann_file = dataset_dict['FoggyCityscape_amodal_val']['ann_file']
            else:
                cfg.test_dataloader.dataset.ann_file = dataset_dict['FoggyCityscape_modal_val']['ann_file']
            cfg.test_evaluator['ann_file'] = os.path.join(cfg.data_root, 'leftImg8bit_foggyDBF/', cfg.test_dataloader.dataset.ann_file)
            
            
            
            
    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)
    if args.tta:
        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    
    # cfg.env_cfg.mp_cfg
    # build the runner from config
    if args.runner: 
        if args.runner == 'Runner_Modified':
            cfg.no_display = args.no_display
            runner = Runner_Modified.from_cfg(cfg)
        if args.runner == 'Runner_Dump':
            cfg.no_display = args.no_display
            runner = Runner_Dump.from_cfg(cfg)
            runner.start_index = args.start_index
            runner.end_index = args.end_index

    elif 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)



    if args.flop_analysis:
        print(f" Model parameters: {np.sum([int(np.prod(p.shape)) for p in runner.model.parameters()]):,}")
        runner.model.backbone
        import pdb
        pdb.set_trace()
        quit()
        
    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
