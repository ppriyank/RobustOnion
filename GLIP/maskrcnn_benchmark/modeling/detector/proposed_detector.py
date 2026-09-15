import logging
import torch 
import torch.nn.functional as F
import copy 
from einops import rearrange, repeat
from .generalized_vl_rcnn import GeneralizedVLRCNN


from maskrcnn_benchmark.data.datasets.perturb_util import normalize, save_image
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.engine.utils import save_img, display_box_over_img


class GeneralizedVLRCNN_PROPOSE(GeneralizedVLRCNN):
    def __init__(self, cfg=None):
        backbone_type = cfg.MODEL.BACKBONE_TYPE
        backbone_switch_off = True  
        cfg.defrost()
        if (backbone_type) and (backbone_type != 'VANIL'):
            cfg.MODEL.BACKBONE.CONV_BODY = cfg.MODEL.BACKBONE.CONV_BODY + "-" +backbone_type
            backbone_switch_off = False 

        if not cfg.MODEL.E2E:
            cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True 
            cfg.MODEL.FPN.FREEZE = True 
            cfg.MODEL.RPN.FREEZE = True 
        else:
            if cfg.MODEL.E2E_MODE and 'rpn' in cfg.MODEL.E2E_MODE:
                cfg.MODEL.LANGUAGE_BACKBONE.FREEZE = True 
                cfg.MODEL.FPN.FREEZE = True 
                cfg.MODEL.RPN.FREEZE = False  
            else:
                backbone_switch_off = False   

            
        cfg.freeze()

        super().__init__(cfg=cfg)
        if not cfg.MODEL.E2E or cfg.MODEL.E2E_MODE:
            self.switch_off_existing_modules(switch_backbone= backbone_switch_off)
        
        
        for name, p in self.named_parameters():
            if p.requires_grad == True:
                print (" **** " , name, p.data.shape)
        
        
    def switch_off_existing_modules(self, switch_backbone=None, pred_boxes=None):
        if switch_backbone:
            for p in self.backbone.body.parameters():p.requires_grad = False
            self.freeze_backbone = True 
        else:
            self.freeze_backbone = False

        if self.cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.language_backbone.body.parameters():
                p.requires_grad = False
            self.freeze_language_backbone = True 

        if pred_boxes:
            for name, param in self.rpn.named_parameters():
                if "scales" in name or "bbox_pred" in name or "centerness" in name or "prompts" in name:continue
                param.requires_grad = False
        elif 'rpn' in self.cfg.MODEL.E2E_MODE:
            if 'dyhead_tower' in self.cfg.MODEL.E2E_MODE:
                for name, param in self.rpn.named_parameters():
                    if "dyhead_tower" in name:continue
                    param.requires_grad = False
        else:
            for p in self.rpn.parameters():
                p.requires_grad = False
        self.freeze_rpn = self.cfg.MODEL.RPN.EXPLICIT_FREEZE
        
        if self.roi_heads is not None:
            for p in self.roi_heads.parameters():
                p.requires_grad = False
            
        if self.cfg.MODEL.FPN.FREEZE:
            if hasattr(self.backbone, 'fpn'):
                for p in self.backbone.fpn.parameters():
                    p.requires_grad = False
                # self.freeze_fpn = cfg.MODEL.FPN.FREEZE
                self.freeze_fpn = True 

        if hasattr(self.rpn.head, 'cls_logits'):
            for p in self.rpn.head.cls_logits.parameters():
                p.requires_grad = False
            # self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
            self.freeze_cls_logits = True 
        
        


        # self.add_linear_layer
        
    def custom_load(self):
        logger = logging.getLogger(__name__)
        logger.info("\n\n CUSTOM LOADING OF ADDITIONAL PARAMETERS \n\n")
        
    def switch_off_modules(self):
        assert len(self.backbone) == 2 , "Backbone has modules more than backbone + neck"
        if hasattr(self.backbone[0], "swith_off_added_modules"):
            self.backbone[0].swith_off_added_modules()
    
    def switch_on_modules(self):
        assert len(self.backbone) == 2, "Backbone has modules more than backbone + neck"
        if hasattr(self.backbone[0], "swith_on_added_modules"):
            self.backbone[0].swith_on_added_modules()

class GeneralizedVLRCNN_TRILANG(GeneralizedVLRCNN_PROPOSE):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.N_LR = cfg.DATASETS.N_LR
        # no of captions per sampels
        # 2 ==> 2 captions per image 


        self.N_BR = cfg.DATASETS.N_BR 
        # no of samples Low Quality Samples in a batch 
        # 2 ==> 1 HQ and 1 LQ 

        # Total Embeeddings :: self.N_LR * self.N_BR * BATCH SIZE
        # (In addition to original emebddings )
        # * BATCH SIZE :: IMAGE _ TEXT PAIR 

        self.noisy_text_emebdding = True 
        
    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None, new_captions=None):
        images = to_image_list(images)
        device = images.tensors.device

        # save_image(normalize(images.tensors), "test.png")
        language_dict_features_new = None     
        with torch.no_grad():
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                input_ids = tokenized.input_ids
                mlm_labels = None
                tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.language_backbone(tokenizer_input)
                language_dict_features["mlm_labels"] = mlm_labels
                
                if new_captions is not None:
                    assert self.training == True, "new captions valid only during training" 
                    new_captions_list = []
                    N_LR = len(new_captions[0])
                    assert self.N_LR * self.N_BR == N_LR, "New captions not sufficient for each input image....."
                    for e in new_captions:
                        new_captions_list += e
                    # for e in new_captions_list:print(e)

                    tokenized = self.tokenizer.batch_encode_plus(new_captions_list, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                    input_ids = tokenized.input_ids
                    tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                    language_dict_features_new = self.language_backbone(tokenizer_input)
                    language_dict_features_new["mlm_labels"] = mlm_labels
                elif self.training and self.noisy_text_emebdding:
                    N_LR = self.N_LR * self.N_BR
                    language_dict_features_new = copy.deepcopy(language_dict_features)
                    
                    language_dict_features_new['aggregate'] = repeat(language_dict_features_new['aggregate'], "B C -> (B N) C", N=self.N_LR)
                    noise = torch.normal(0, 1, size=language_dict_features_new['aggregate'].shape).to( language_dict_features_new['aggregate'].device ) 
                    language_dict_features_new['aggregate'] += noise
                    # language_dict_features['aggregate']
                    
                    language_dict_features_new['embedded'] = repeat(language_dict_features_new['embedded'], "B D C -> (B N) D C", N=self.N_LR)
                    noise = torch.normal(0, 1, size=language_dict_features_new['embedded'].shape).to( language_dict_features_new['embedded'].device ) 
                    language_dict_features_new['embedded'] += noise
                    # language_dict_features_new['embedded'].mean(1)
                    # language_dict_features['embedded'].mean(1)

                    language_dict_features_new['masks'] = repeat(language_dict_features_new['masks'], "B C -> (B N) C", N=self.N_LR)

                    language_dict_features_new['hidden'] = repeat(language_dict_features_new['hidden'], "B D C -> (B N) D C", N=self.N_LR)
                    noise = torch.normal(0, 1, size=language_dict_features_new['hidden'].shape).to( language_dict_features_new['hidden'].device )
                    language_dict_features_new['hidden'] += noise
                    # language_dict_features['hidden']
                    
                            
        # visual embedding
        swint_feature_c4 = None
        visual_features = self.backbone(images.tensors)

        # rpn force boxes
        if targets: targets = [target.to(device) for target in targets if target is not None]
        proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, language_dict_features_new=language_dict_features_new)

        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

class GeneralizedVLRCNN_DISTILL(GeneralizedVLRCNN_PROPOSE):
        
    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None, new_captions=None):
        images = to_image_list(images)
        device = images.tensors.device

        
        language_dict_features_new = None     
        with torch.no_grad():
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                input_ids = tokenized.input_ids
                mlm_labels = None
                tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.language_backbone(tokenizer_input)
                language_dict_features["mlm_labels"] = mlm_labels

                if new_captions is not None:
                    assert self.training == True, "new captions valid only during training" 
                    new_captions_list = []
                    N_LR = len(new_captions[0])
                    for e in new_captions:
                        new_captions_list += e
                    # for e in new_captions_list:print(e)
                    
                    
                    tokenized = self.tokenizer.batch_encode_plus(new_captions_list, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                    input_ids = tokenized.input_ids
                    tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                    language_dict_features_new = self.language_backbone(tokenizer_input)
                    language_dict_features_new["mlm_labels"] = mlm_labels
                
        visual_features_HD_teacher = None 
        if self.training:
            with torch.no_grad():
                # save_image(normalize(images.tensors[::2]), "test_HD.png")
                self.switch_off_modules()
                self.backbone.eval()
                # print( self.backbone(images.tensors[::2])[-1].mean(1).sum(-1) ) 
                # print( self.backbone(images.tensors[::2])[-1].mean(1).sum(-1) ) 
                # print( self.backbone(images.tensors[::2])[-1].mean(1).sum(-1) ) 
                visual_features_HD_teacher = self.backbone(images.tensors[::2])
                self.backbone.train()
                self.switch_on_modules()
            
        
        # visual embedding
        swint_feature_c4 = None
        visual_features = self.backbone(images.tensors)
        # visual_features[-1].mean(1).sum(-1)
        # print( self.backbone(images.tensors[::2])[-1].mean(1).sum(-1) ) 
        # print( self.backbone(images.tensors)[-1].mean(1).sum(-1) ) 

        # rpn force boxes
        if targets: targets = [target.to(device) for target in targets if target is not None]
        proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, language_dict_features_new=language_dict_features_new,
        visual_features_HD_teacher=visual_features_HD_teacher)

        
        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

class GeneralizedVLRCNN_VAR(GeneralizedVLRCNN_PROPOSE):
        
    def forward(self,  images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None):
        images = to_image_list(images)
        device = images.tensors.device

        language_dict_features = {}
        if captions is not None:
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)
            input_ids = tokenized.input_ids
            mlm_labels = None
            tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}

            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)

            language_dict_features["mlm_labels"] = mlm_labels

        # visual embedding
        swint_feature_c4 = None
        visual_features = self.backbone(images.tensors)

        # rpn force boxes
        if targets:
            targets = [target.to(device) for target in targets if target is not None]

        proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4)
        
        x = rearrange(visual_features[-1] , "B C H W -> B C (H W)")
        x = x.mean(-1)
        x = F.normalize(x, p=2, dim=-1,)
        var_loss = -((x.mean(0) - x)** 2).sum()
        var_loss = (var_loss.exp() + 1).log()
        
        x = visual_features
        result = proposals
        detector_losses = {'var': var_loss}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result


class GeneralizedVLRCNN_RL(GeneralizedVLRCNN_TRILANG):
        
    def forward_feats(self, images):
        images = to_image_list(images)
        visual_features = self.backbone(images.tensors)
        return rearrange(visual_features[0], 'B C H W -> B C (H W)').mean(-1)

    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None, new_captions=None, suffix=''):
        images = to_image_list(images)
        device = images.tensors.device

        language_dict_features_new = None     
        with torch.no_grad():
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                input_ids = tokenized.input_ids
                mlm_labels = None
                tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.language_backbone(tokenizer_input)
                language_dict_features["mlm_labels"] = mlm_labels

                if new_captions is not None:
                    assert self.training == True, "new captions valid only during training" 
                    new_captions_list = []
                    N_LR = len(new_captions[0])
                    for e in new_captions:
                        new_captions_list += e
                    # for e in new_captions_list:print(e)
                    
                    tokenized = self.tokenizer.batch_encode_plus(new_captions_list, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                    input_ids = tokenized.input_ids
                    tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                    language_dict_features_new = self.language_backbone(tokenizer_input)
                    language_dict_features_new["mlm_labels"] = mlm_labels

        # save_img(images.tensors, "temp.png")
        # visual embedding
        swint_feature_c4 = None
        visual_features = self.backbone(images.tensors)

        # rpn force boxes
        if targets: targets = [target.to(device) for target in targets if target is not None]
        z = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, language_dict_features_new=language_dict_features_new)
        (proposals, proposal_losses, used_visual_features), variant_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, language_dict_features_new=language_dict_features_new)

        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if suffix:
                losses = {e+"_"+suffix:losses[e] for e in losses}
            return losses, variant_visual_features

        return result
        
        




class GeneralizedVLRCNN_RobustSAM(GeneralizedVLRCNN_PROPOSE):
    def __init__(self, cfg=None):
        cfg.defrost()
        cfg.MODEL.E2E = False 
        cfg.MODEL.RPN_ARCHITECTURE = 'VLDyHeadModule_RobustSAM'
        cfg.freeze()
        super().__init__(cfg=cfg,)

        cfg.defrost()
        cfg.MODEL.RPN.FREEZE = False 
        cfg.freeze()
        
        self.backbone.fpn.return_swint_feature_before_fusion = True 
        
        assert cfg.DATASETS.REPLICA_END == True 
        # assert self.cfg.DATASETS.N_LR == 2 

        # self.N_LR = self.cfg.DATASETS.N_LR
        self.B_LR = cfg.DATASETS.N_BR - 1
        # pair of images, 0 --> HQ 1 --> Noisy 

        for name, p in self.named_parameters():
            if p.requires_grad == True:
                print (" **** " , name, p.data.shape)
        

        B = cfg.SOLVER.IMS_PER_BATCH // cfg.num_gpus
        clear_labels = ([1] + ([0] * self.B_LR) ) * int(B)
        if cfg.DATASETS.REPLICA_END:
            clear_labels = [1] * int(B) + [0] * self.B_LR  * int(B)
        self.clear_labels = torch.tensor(clear_labels)
        # .to(device)
        
        


    def switch_off_existing_modules(self, switch_backbone=None, pred_boxes=None):
        self.freeze_backbone = False
        for p in self.language_backbone.body.parameters():
            p.requires_grad = False
        self.freeze_language_backbone = True 
        
        for name, param in self.rpn.named_parameters():
            if "robust" not in name:
                param.requires_grad = False
        self.freeze_rpn = self.cfg.MODEL.RPN.EXPLICIT_FREEZE
        

        for p in self.backbone.fpn.parameters():
                p.requires_grad = False
        self.freeze_fpn = True 
        
        for p in self.rpn.head.cls_logits.parameters():
                p.requires_grad = False
        self.freeze_cls_logits = True 
            
    
    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None):
        
        images = to_image_list(images)
        device = images.tensors.device
        # save_img(images.tensors, "test.png")
        # targets[0].__dict__.keys()
        
        # if self.cfg.DATASETS.REPLICA_END:
        #     display_box_over_img(images.tensors, targets, border_width = 2, prefix_name="temp2", mode="all", grid_shape= int(self.cfg.SOLVER.IMS_PER_BATCH // self.cfg.num_gpus) )
        # else:
        #     display_box_over_img(images.tensors, targets, border_width = 2, prefix_name="temp2", mode="all", grid_shape= int(self.B_LR + 1) )
        
        # language embedding
        language_dict_features = {}
        if captions is not None:
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)
            input_ids = tokenized.input_ids
            mlm_labels = None    
            tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
        
            language_dict_features["mlm_labels"] = mlm_labels

        # visual embedding
        swint_feature_c4 = None
        visual_features, _, shallow_feats = self.backbone(images.tensors)
        
        # [e.shape for e in visual_features]
        # [torch.Size([30, 256, 52, 168]), torch.Size([30, 256, 26, 84]), torch.Size([30, 256, 13, 42]), torch.Size([30, 256, 7, 21]), torch.Size([30, 256, 4, 11])]
        # shallow_feats
        # torch.Size([30, 96, 104, 336])        

        # rpn force boxes
        if targets:
            targets = [target.to(device) for target in targets if target is not None]

        proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, clear_labels=self.clear_labels, shallow_feats=shallow_feats)
        
        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

class GeneralizedVLRCNN_TRILANG_DEBUG(GeneralizedVLRCNN_TRILANG):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        for name, p in self.named_parameters():
            if 'norm3' not in name:
                p.requires_grad = False
                
    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None, new_captions=None):
        images = to_image_list(images)
        device = images.tensors.device
        
        # save_image(normalize(images.tensors), "test.png")
        language_dict_features_new = None     
        with torch.no_grad():
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                input_ids = tokenized.input_ids
                mlm_labels = None
                tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.language_backbone(tokenizer_input)
                language_dict_features["mlm_labels"] = mlm_labels

                if new_captions is not None:
                    assert self.training == True, "new captions valid only during training" 
                    new_captions_list = []
                    N_LR = len(new_captions[0])
                    for e in new_captions:
                        new_captions_list += e
                    # for e in new_captions_list:print(e)
                    
                    
                    tokenized = self.tokenizer.batch_encode_plus(new_captions_list, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                    input_ids = tokenized.input_ids
                    tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                    language_dict_features_new = self.language_backbone(tokenizer_input)
                    language_dict_features_new["mlm_labels"] = mlm_labels
        
        # save_image(normalize(images.tensors), "test.png")
        with torch.no_grad():
            # visual embedding
            swint_feature_c4 = None
            visual_features = self.backbone(images.tensors)

        
        visual_features[-1]
        import pdb
        pdb.set_trace()
        
        # rpn force boxes
        if targets: targets = [target.to(device) for target in targets if target is not None]
        proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, language_dict_features_new=language_dict_features_new)

        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

class GeneralizedVLRCNN_TRILANG_DUMP(GeneralizedVLRCNN_PROPOSE):
                
    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None, new_captions=None, mean=True, true_backbone=False, only_last=True ):
        images = to_image_list(images)        
        # save_image(normalize(images.tensors), "test.png")
        with torch.no_grad():
            # visual embedding
            swint_feature_c4 = None
            if true_backbone:
                visual_features = self.backbone[0](images.tensors)
            else:
                visual_features = self.backbone(images.tensors)

        
        if mean:
            if only_last:
                return rearrange(visual_features[-2], "B C H W -> B C (H W)").mean(-1)
            return [rearrange(e, "B C H W -> B C (H W)").mean(-1) for e in visual_features]
        else:
            # [e.shape for e in visual_features]
            # [torch.Size([1, 256, 100, 152]), torch.Size([1, 256, 50, 76]), torch.Size([1, 256, 25, 38]), torch.Size([1, 256, 13, 19]), torch.Size([1, 256, 7, 10])]
            return [rearrange(e, "B C H W -> B C (H W)") for e in visual_features]
        
class GeneralizedVLRCNN_TRILANG_GFLOP(GeneralizedVLRCNN_TRILANG):
    
    def forward(self, images,  targets=None,  captions=None,  positive_map=None, greenlight_map=None, new_captions=None):
        import pdb
        pdb.set_trace()
        # images.tensors
        # torch.Size([2, 3, 736, 960])
        # images.image_sizes
        # [torch.Size([720, 954]), torch.Size([720, 954])]
        # targets
        # [BoxList(num_boxes=2, image_width=954, image_height=720, mode=xyxy), BoxList(num_boxes=2, image_width=954, image_height=720, mode=xyxy)]
        
        # targets[0].bbox
        # tensor([[263.3870, 325.7637, 294.4957, 352.7378], [288.2739, 269.7406, 723.7957, 454.4092]], device='cuda:0')
        # targets[0].size (954, 720)
        # targets[0].mode = 'xyxy'
        # targets[0].extra_fields.keys()
        # targets[0].extra_fields['labels'] tensor([1, 1], device='cuda:0')
        # targets[0].extra_fields['boxes'] tensor([[127., 157., 142., 170.], [139., 130., 349., 219.]], device='cuda:0')
        # targets[0].extra_fields['caption'] 'A black and yellow bird is eating a worm while resting on the grassy ground .'
        # targets[0].extra_fields['image_id'] tensor([810], device='cuda:0')
        # targets[0].extra_fields['tokens_positive'] = [[[34, 40]], [[0, 23]]]
        # targets[0].extra_fields['area'] = tensor([  224., 18990.], device='cuda:0')
        # targets[0].extra_fields['iscrowd'] = tensor([0, 0], device='cuda:0')
        # targets[0].extra_fields['orig_size'] = tensor([347, 460], device='cuda:0')
        # targets[0].extra_fields['size'] = tensor([347, 460], device='cuda:0')
        # targets[0].extra_fields['positive_map'] = tensor([347, 460], device='cuda:0')
        # 'size', 'positive_map', 'greenlight_map', 'positive_map_for_od_labels', 'original_od_label', 'dataset_name', 'sentence_id', 'original_img_id', 'new_captions'])
        # targets[0].extra_fields
        
        images = to_image_list(images)
        device = images.tensors.device
        
        # save_image(normalize(images.tensors), "test.png")
        language_dict_features_new = None     
        with torch.no_grad():
            # language embedding
            language_dict_features = {}
            if captions is not None:
                #print(captions[0])
                tokenized = self.tokenizer.batch_encode_plus(captions, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                input_ids = tokenized.input_ids
                mlm_labels = None
                tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                language_dict_features = self.language_backbone(tokenizer_input)
                language_dict_features["mlm_labels"] = mlm_labels

                if new_captions is not None:
                    assert self.training == True, "new captions valid only during training" 
                    new_captions_list = []
                    N_LR = len(new_captions[0])
                    for e in new_captions:
                        new_captions_list += e
                    # for e in new_captions_list:print(e)
                    
                    
                    tokenized = self.tokenizer.batch_encode_plus(new_captions_list, max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                            return_special_tokens_mask=True, return_tensors='pt', truncation=True).to(device)
                
                    input_ids = tokenized.input_ids
                    tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
                    language_dict_features_new = self.language_backbone(tokenizer_input)
                    language_dict_features_new["mlm_labels"] = mlm_labels
        
        # save_image(normalize(images.tensors), "test.png")
        with torch.no_grad():
            # visual embedding
            swint_feature_c4 = None
            visual_features = self.backbone(images.tensors)

        
        visual_features[-1]
        import pdb
        pdb.set_trace()
        
        # rpn force boxes
        if targets: targets = [target.to(device) for target in targets if target is not None]
        proposals, proposal_losses, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4, language_dict_features_new=language_dict_features_new)

        # RPN-only models don't have roi_heads
        x = visual_features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result




###### Dump all features 
class GeneralizedVLRCNN_DUMPING(GeneralizedVLRCNN):
    def __init__(self, cfg=None, **kwargs):
        cfg.defrost()
        original_name = cfg.MODEL.RPN_ARCHITECTURE
        cfg.MODEL.RPN_ARCHITECTURE = "VLDyHeadModule_Dump"
        cfg.MODEL.RPN.RETURN_FUSED_FEATURES = True
        cfg.freeze()
        super().__init__(cfg=cfg, **kwargs)

        cfg.defrost()
        cfg.MODEL.RPN_ARCHITECTURE = original_name
        cfg.freeze()


    def forward(self, 
        images, 
        targets=None, 
        captions=None, 
        positive_map=None,
        greenlight_map=None,
        new_captions=None):
        
        images = to_image_list(images)
        device = images.tensors.device


        # language embedding
        language_dict_features = {}
        if captions is not None:
            #print(captions[0])
            tokenized = self.tokenizer.batch_encode_plus(captions,
                                                        max_length=self.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
                                                        padding='max_length' if self.cfg.MODEL.LANGUAGE_BACKBONE.PAD_MAX else "longest",
                                                        return_special_tokens_mask=True,
                                                        return_tensors='pt',
                                                        truncation=True).to(device)
            input_ids = tokenized.input_ids
            mlm_labels = None
        
            
            tokenizer_input = {"input_ids": input_ids, "attention_mask": tokenized.attention_mask}
            with torch.no_grad():
                language_dict_features = self.language_backbone(tokenizer_input)
            
            language_dict_features["mlm_labels"] = mlm_labels


        # save_image(normalize(images.tensors), "test.png")
        # quit()
        
        # visual embedding
        swint_feature_c4 = None
        body_features = self.backbone[0](images.tensors)
        # [e.shape for e in body_features]
        # [torch.Size([1, 96, 200, 304]), torch.Size([1, 192, 100, 152]), torch.Size([1, 384, 50, 76]), torch.Size([1, 768, 25, 38])]
        
        body_features= [rearrange(e, "B C H W -> B C (H W)").mean(-1).cpu().numpy() for e in body_features]
        # [e.shape for e in body_features]
        # [(1, 96), (1, 192), (1, 384), (1, 768)]
        # x[0], x[1], x[2], x[3]

        visual_features = self.backbone(images.tensors)
        # print([e.shape for e in visual_features])
        # [torch.Size([1, 256, 100, 152]), torch.Size([1, 256, 50, 76]), torch.Size([1, 256, 25, 38]), torch.Size([1, 256, 13, 19]), torch.Size([1, 256, 7, 10])]

    
        neck_features = rearrange(visual_features[3], "B C H W -> B C (H W)").mean(-1).cpu().numpy()
        # (1, 256)
        
        # rpn force boxes
        if targets:targets = [target.to(device) for target in targets if target is not None]
        
        boxes, fused_visual_features = self.rpn(images, visual_features, targets, language_dict_features, positive_map, captions, swint_feature_c4)
        # print([e.shape for e in fused_visual_features])
        # [torch.Size([1, 256, 100, 152]), torch.Size([1, 256, 50, 76]), torch.Size([1, 256, 25, 38]), torch.Size([1, 256, 13, 19]), torch.Size([1, 256, 7, 10])]

        fused_visual_features = rearrange(fused_visual_features[3], "B C H W -> B C (H W)").mean(-1).cpu().numpy()

        
        result = dict(backbone=body_features, backbone_fpn=neck_features, fused_visual_features=fused_visual_features)
        return result  
    




###### Compare all feats 
import torch
from torch import nn
from ..backbone import build_backbone
from maskrcnn_benchmark.modeling import registry
# from ..backbone import Siamese_TF

class Feat_Comp(nn.Module):
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        model = registry.BACKBONES[cfg.MODEL.BACKBONE]
        self.backbone = model(
            img_size=cfg.MODEL.SIZE, factor=cfg.MODEL.FACTOR, no_of_points=cfg.MODEL.SAMPLE, loss_type=cfg.MODEL.LOSS,
            load_pretrain=cfg.MODEL.PRETRAIN, enable_dropout=cfg.MODEL.DROPOUT, 
        )
    
    def forward(self, f1, f2, labels=None):
        if self.training:
            return self.backbone(f1, f2, labels)
        else:
            return self.backbone(f1, f2)
        
