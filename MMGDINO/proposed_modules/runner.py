
import copy 
import torch 
import os 
import os.path as osp
import pickle

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from functools import partial
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.loops import TestLoop
from mmengine.runner.amp import autocast
from mmengine.runner import Runner

from mmengine.utils.dl_utils import collect_env          
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)

from mmengine.logging import MMLogger
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)

from mmengine.runner.log_processor import LogProcessor
from mmengine.hooks import Hook

from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm, is_model_wrapper, revert_sync_batchnorm)
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler, build_optim_wrapper)
from mmengine.config import Config, ConfigDict

from mmengine.dist import get_dist_info

import numpy as np 


# https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py
@RUNNERS.register_module()
class Runner_Modified(Runner):
    def __init__(self, cfg=None, **kwargs):
        self.no_display  = cfg.no_display
        super().__init__(cfg=cfg, **kwargs)
        
         
    def build_logger(self,
                     log_level: Union[int, str] = 'INFO',
                     log_file: Optional[str] = None,
                     **kwargs) -> MMLogger:
        
        
        self._log_dir = self.work_dir
        if log_file is None:
            log_file = osp.join(self._log_dir, f'{self.timestamp}.log')

        log_cfg = dict(log_level=log_level, log_file=log_file, **kwargs)
        log_cfg.setdefault('name', self._experiment_name)
        log_cfg.setdefault('file_mode', 'a')
        return MMLogger.get_instance(**log_cfg)  # type: ignore

    def _log_env(self, env_cfg: dict) -> None:
        rank, world_size = get_dist_info()
        # Collect and log environment information.
        env = collect_env()
        if self.cfg._cfg_dict and (not self.no_display) and rank == 0 :
            print(self.cfg._cfg_dict)
            # self.logger.info(f'Config:\n{self.cfg.pretty_text}')
        
    def get_hooks_info(self) -> str:
        _ = 0 
        '''
        do nothing 
        '''

    def test(self) -> dict:
        # mmengine.runner.loops.TestLoop
        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # self.dummy_test()
        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')

        
        return metrics
    
    def dummy_test(self):
        import pdb
        pdb.set_trace()
        # https://mmengine.readthedocs.io/en/v0.9.0/_modules/mmengine/runner/loops.html#TestLoop
        Evaluator = self._test_loop.evaluator
        hasattr(self._test_loop.dataloader.dataset, 'metainfo')
        Evaluator.dataset_meta

        
        self._test_loop.runner.call_hook('before_test')
        self._test_loop.runner.call_hook('before_test_epoch')
        self._test_loop.runner.model.eval()
        for idx, data_batch in enumerate(self._test_loop.dataloader):
            self._test_loop.run_iter(idx, data_batch)

        # outputs = self.model.test_step(data_batch)
        # self._test_loop.evaluator.process(data_samples=outputs, data_batch=data_batch)

        # self._test_loop.runner.call_hook
        # 'after_test_iter'
        # https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/evaluator.py
        # mmengine.evaluator.evaluator.Evaluator 
        # self._test_loop.evaluator

        metrics = self._test_loop.evaluator.evaluate(len(self._test_loop.dataloader.dataset))

        
        
    


    
    











# from mmengine.runner.runner import Runner

@LOOPS.register_module()
class TestLoop_Dump(TestLoop):
    def __init__(self, dump_index=None, backbone_only=None, **kwargs):
        super().__init__(**kwargs)
        self.dump_index = dump_index
        self.backbone_only = backbone_only 
        if backbone_only:
            self.runner.model.backbone_only = True 
            self.runner.model.backbone.out_indices = (0, 1, 2, 3)
        
    def run(self) -> dict:
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        # clear test loss
        self.test_loss.clear()

        BACKBONE = []
        BOXES = []
        NECK =[]
        ENCODER = []
        DECODER = []

        start_index = self.runner.start_index
        end_index = self.runner.end_index

        N = len(self.dataloader)
        for idx, data_batch in enumerate(self.dataloader):
            print( f"{idx} /{N}", end="\r" )
            if end_index and idx > end_index:break 
            if (start_index != None and end_index != None) and (idx < start_index or idx > end_index):continue 
            
            output = self.run_iter(idx, data_batch)

            BACKBONE.append(output['BACKBONE'])
            NECK.append(output.get('NECK', None))
            ENCODER.append(output.get('ENCODER', None))
        
        if not self.backbone_only:
            FUSED_FEATS = [[], []]
            for e in ENCODER:
                assert len(e) == 1
                FUSED_FEATS[0].append(e[0][0])
                FUSED_FEATS[1].append(e[0][1])
            FUSED_FEATS[0] = np.concatenate(FUSED_FEATS[0])
            FUSED_FEATS[1] = np.concatenate(FUSED_FEATS[1])
            # [e.shape for e in FUSED_FEATS]
            # [(5, 256), (5, 256)]

            NECK = np.concatenate(NECK, 0)
            assert NECK.shape[1] == 1
            NECK = NECK[:,0]
            # NECK.shape
            # (5, 1, 256)

        if self.dump_index is not None:
            BACKBONE_LAYERS = [[] for i in self.dump_index]
            for i,layer in enumerate(BACKBONE):
                assert len(layer) == 1
                for k in self.dump_index:
                    BACKBONE_LAYERS[k].append(layer[0][k])
            
            for k in self.dump_index:
                BACKBONE_LAYERS[k] = np.concatenate(BACKBONE_LAYERS[k])
        else:
            BACKBONE_LAYERS = [[] for i in range(4)]
            for i,layer in enumerate(BACKBONE):
                assert len(layer) == 1
                BACKBONE_LAYERS[0].append(layer[0][0])
                BACKBONE_LAYERS[1].append(layer[0][1])
                BACKBONE_LAYERS[2].append(layer[0][2])
                BACKBONE_LAYERS[3].append(layer[0][3])

            BACKBONE_LAYERS[0] = np.concatenate(BACKBONE_LAYERS[0])
            BACKBONE_LAYERS[1] = np.concatenate(BACKBONE_LAYERS[1])
            BACKBONE_LAYERS[2] = np.concatenate(BACKBONE_LAYERS[2])
            BACKBONE_LAYERS[3] = np.concatenate(BACKBONE_LAYERS[3])
            # [e.shape for e in BACKBONE_LAYERS]
            # [(5, 192), (5, 384), (5, 768), (5, 1536)]

        if self.backbone_only:
            to_be_dumped = {"BACKBONE":  BACKBONE_LAYERS}
        else:
            to_be_dumped = {
                "BACKBONE":  BACKBONE_LAYERS, 
                "NECK": NECK, 
                "FUSED_FEATS": FUSED_FEATS, 
            }
        
        if start_index and end_index:
            # save_image(data_batch['inputs'][0] / 256, "temp.png")
            # pickle_name = os.path.join(self.runner._work_dir, f"predictions_{start_index}_{end_index}-{self.runner.sev}")
            pickle_name = os.path.join(self.runner._work_dir, f"predictions_{start_index}_{end_index}")
        else:
            pickle_name = os.path.join(self.runner._work_dir, f"predictions")

        with open(f'{pickle_name}.pkl', 'wb') as handle:
                pickle.dump(to_be_dumped, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return 

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        self.runner.call_hook('before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
        return outputs
        
        

@RUNNERS.register_module()
class Runner_Dump(Runner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.start_index = None
        self.end_index = None 
        self.sev = None

    def build_test_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        
        loop_cfg = copy.deepcopy(loop)  # type: ignore
        loop_cfg['type'] = 'TestLoop_Dump' 
        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict( runner=self, dataloader=self._test_dataloader, evaluator=self._test_evaluator))
        
        return loop  # type: ignore

    def test(self) -> dict:
        
        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        return 

