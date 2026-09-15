

import time
import os 
import torch
from torch import nn
import logging
import pickle
import numpy as np 
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import EvalHook, BestCheckpointer, PeriodicCheckpointer
# from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from contextlib import ExitStack, contextmanager
from detectron2.utils.comm import get_world_size, is_main_process   
from detectron2.evaluation.evaluator import inference_context 



from torchvision.utils import save_image
def save_img(x, name="temp.png", **kwargs ):
    save_image( normalize( x ), name, **kwargs )
    
def normalize(x):return (x - x.min()) / (x.max() - x.min())


def inference_on_dataset_dump(model, data_loader, log_every_sec=5, start_index=1, end_index=5000, 
    sev=-1, output_folder=None, dump_index=None, dump_mode=None):
    logger = logging.getLogger(__name__)

    num_devices = get_world_size()
    assert num_devices == 1
    
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    
    BACKBONE = []
    FUSED_FEATS = []
    BACKBONE_FPN = []
    FILE_NAME = []
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            assert len(inputs) == 1
            if (start_index is not None and end_index is not None ) and (idx < start_index or idx > end_index):
                continue 
            print(idx, "/", len(data_loader), end="\r")
            # len(inputs), inputs[0]['image'].shape
            # save_img(inputs[0]['image'].float())
            
            outputs = model(inputs)
            if dump_mode == 'backbone_only':
                backbone, _, _ = outputs
                # print([e.shape for e in backbone])
                if dump_index is not None:
                    BACKBONE.append(backbone[dump_index])
                else:
                    BACKBONE.append(backbone)
            else:
                # outputs
                backbone, neck, encoder = outputs
                # print([e.shape for e in backbone])
                # [(1, 1024), (1, 1024), (1, 1024), (1, 1024)]

                # print([e.shape for e in neck])
                # [(1, 256), (1, 256)]

                # print([e.shape for e in encoder])
                # [(1, 256), (1, 256)]

                BACKBONE.append(backbone)
                FUSED_FEATS.append(encoder)
                BACKBONE_FPN.append(neck)
            FILE_NAME += [e['file_name'] for e in inputs]

    
    # [e.shape for e in BACKBONE]    
    if dump_mode == 'backbone_only':
        to_be_dumped = {"BACKBONE":  BACKBONE}
        print(np.concatenate(BACKBONE,0).shape)
    else:
        to_be_dumped = {"BACKBONE":  BACKBONE,  "BACKBONE_FPN" : BACKBONE_FPN,  "FUSED_FEATS": FUSED_FEATS,  "FILE_NAME": FILE_NAME , }

    if start_index and end_index:
        pickle_name = os.path.join(output_folder, f"predictions_{start_index}_{end_index}-{sev}")
    else:
        pickle_name = os.path.join(output_folder, f"predictions")
    with open(f'{pickle_name}.pkl', 'wb') as handle:
        pickle.dump(to_be_dumped, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    
    




class DefaultTrainer_Custom(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # self._trainer
        # <detectron2.engine.train_loop.AMPTrainer object at 0x736c39ee6250>

        self.sanity_check = cfg.SANITY_CHECK
        # for e in self._hooks: print(e)

        if "LORA" in cfg.MODEL.BACKBONE.NAME:
            self.attention_fix_applied = True 

    def train(self):
        if self.sanity_check:
            self.test(self.cfg, self.model)
                    
        _last_eval_results = super().train()
        return _last_eval_results
        
        # if comm.is_main_process():
            # self._checkpointer.save(f"{self._file_prefix}", **_last_eval_results)    
            # return _last_eval_results

    @classmethod
    def test(cls, cfg, model, evaluators=None, log_every_sec=500):
        return super().test(cfg=cfg, model=model, evaluators=evaluators, log_every_sec=log_every_sec)


    @classmethod
    def test_DUMP(cls, cfg, model, evaluators=None, log_every_sec=500, start_index=None, end_index=None):
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            print(f"\n\n ### {dataset_name} \n\n")
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if cfg.TEST.DUMP_MODE == 'backbone_only':
                model.glee.mode = 'backbone_only'

            sev = cfg.SEV
            output_folder= cfg.OUTPUT_DIR
            results_i = inference_on_dataset_dump(model, data_loader, 
                log_every_sec=log_every_sec, start_index=start_index, end_index=end_index, sev=sev, 
                output_folder=output_folder, dump_index=cfg.TEST.DUMP_INDEX, dump_mode=cfg.TEST.DUMP_MODE,
                )
        return None
        


    def resume_or_load(self, resume=True):
        
        # self.checkpointer.load(self.cfg.MODEL.WEIGHTS)
        # ret = self.checkpointer._load_file(self.cfg.MODEL.WEIGHTS)
        # [e for e in ret['model'].keys() if "text_model.embeddings" in e]
        # ['glee.text_encoder.text_model.embeddings.position_ids', 'glee.text_encoder.text_model.embeddings.token_embedding.weight', 'glee.text_encoder.text_model.embeddings.position_embedding.weight']
        # ret['model']['glee.text_encoder.text_model.embeddings.position_ids']

        # glee.text_encoder.text_model.embeddings.position_ids
        # self.model.glee.text_encoder.text_model.embeddings.position_ids

        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = self.iter + 1

        if self.attention_fix_applied:
            # self.model.glee.state_dict().keys()
            # print([e for e in self.model.glee.state_dict().keys() if "attn" in e])

            ret = self.checkpointer._load_file(self.cfg.MODEL.WEIGHTS)
            left_over = dict(model=dict())
            for key in ret["model"]:
                if ("glee.backbone.net.blocks") in key and ("attn") in key and "bias" in key:
                    if "proj" in key : continue

                    # 'backbone.net.blocks.0.attn.q_proj.bias'
                    # 'glee.backbone.net.blocks.0.attn.q_bias'
                    new_key = key.replace("q_bias" , "q_proj.bias").replace("v_bias" , "v_proj.bias").replace("k_bias" , "k_proj.bias")
                    left_over['model'][new_key]= ret["model"][key]


            # for i,blk in enumerate(self.model.glee.backbone.net.blocks):
            #     print("...1", i, blk.attn.q_proj.bias.mean() , blk.attn.v_proj.bias.mean() )
                
            self.checkpointer._load_model(left_over)

            # for i,blk in enumerate(self.model.glee.backbone.net.blocks):
            #     print("...2", i, blk.attn.q_proj.bias.mean() , blk.attn.v_proj.bias.mean() )
            
            
    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        # logger.info("Model:\n{}".format(model))
        
        model_configs = dict(
            matcher=type(model.glee.matcher), 
            backbone= type(model.glee.backbone),
            text_encoder= type(model.glee.text_encoder),
            pixel_decoder= type(model.glee.pixel_decoder),
            predictor= type(model.glee.predictor),
            mask_sptial_embed= type(model.glee.mask_sptial_embed),
            pn_indicator= type(model.glee.pn_indicator),
        )
        
        s = ""
        for key,val in model_configs.items(): s += f"{key} : {val}\n"
        logger.info("\n\nModel:\n{}".format(s))

        return model

    def build_hooks(self, period=500):
        ret = super().build_hooks(period)
        if comm.is_main_process():
            test_eval_period = self.cfg.TEST.EVAL_PERIOD
            val_metric = 'bbox/AP'

            best_eval_hook = BestCheckpointer(
                eval_period=test_eval_period, checkpointer=self.checkpointer, 
                val_metric=val_metric
            )

            N = len(ret)
            i =0 
            while i < N:
                e = ret[i]
                if type(e) == PeriodicCheckpointer:
                    ret.pop(i)
                    continue 
                if type(e) == EvalHook:
                    ret.insert(i+1, best_eval_hook)
                    i +=1
                i +=1

        return ret 
        
        
        
        
        
        