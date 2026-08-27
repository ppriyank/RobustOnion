import bisect
import logging
import time
from typing import Sequence

import torch
from mmengine.logging import print_log
from mmengine.registry import LOOPS

from mmengine.runner.loops import IterBasedTrainLoop, EpochBasedTrainLoop


NONE_IMPORT = ""

from .perturb_util import save_image, normalize, display_boxes
    
        
        




# https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
@LOOPS.register_module()
class IterBasedTrainLoop_Custom(IterBasedTrainLoop):
    def __init__(self,  sanity_check=None,  **kwargs):
        super().__init__(**kwargs)

        # self.max_epochs  1
        # self.max_iters 250000
        # self.epoch 0 
        # self.iter  0 
        # self.val_begin  1
        # self.val_interval 13000

        self.sanity_check = sanity_check
        
    def sanity_check_run(self,):
        self.runner.model.module.switch_off_modules()
        self.runner.val_loop.run()
        self.runner.model.module.switch_on_modules()

    def run(self) -> None:
        if self.sanity_check:
            self.sanity_check_run()
    
        self.runner.call_hook('before_train')
    
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log( f'Advance dataloader {self._iter} steps to skip data ' 
                'that has already been trained', logger='current', level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            # save_image(normalize(data_batch['inputs'][0]), "temp.png")
            # display_boxes(normalize(data_batch['inputs'][0]), data_batch['data_samples'][0].gt_instances.bboxes)

            # for data_batch in self.runner.val_loop.dataloader: break 
            # display_boxes(normalize(data_batch['inputs'][0]), data_batch['data_samples'][0].gt_instances.bboxes)
            # self.runner.cfg.train_cfg.sanity_check
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    
    

    


@LOOPS.register_module()
class EpochBasedTrainLoop_Custom(EpochBasedTrainLoop):
    def __init__(self,  sanity_check=None,  **kwargs):
        super().__init__(**kwargs)

        self.sanity_check = sanity_check

    def sanity_check_run(self,):
        self.runner.model.module.switch_off_modules()
        self.runner.val_loop.run()
        self.runner.model.module.switch_on_modules()


    def run(self) -> torch.nn.Module:
        if self.sanity_check:
            self.sanity_check_run()
    
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and (self._epoch % self.val_interval == 0
                         or self._epoch == self._max_epochs)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

