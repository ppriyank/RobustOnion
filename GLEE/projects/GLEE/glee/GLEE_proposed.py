import torch 
import torch.nn.functional as F

from .utils import box_ops

from detectron2.modeling import META_ARCH_REGISTRY
from .models.proposed_model import GLEE_Model_Proposed, GLEE_Model_DUMP
from .GLEE import GLEE


@META_ARCH_REGISTRY.register()
class GLEE_proposed(GLEE):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        flicker_class_name = ['object']
        self.dataset_name_dicts.update({'flicker':flicker_class_name})
        self.num_class.update({'flicker':1})

        matcher = self.glee.matcher
        
        del self.glee
        self.glee = GLEE_Model_Proposed(cfg, matcher, self.device, self.video_info, cfg.MODEL.CONTRAS_MEAN)

        self.criterion.losses = [e for e in self.criterion.losses if e != 'masks']        
        self.criterion.no_mask_tasks.append("flicker")
        self.criterion.num_classes.update({'flicker' : 1})

        self.all_loss_nams = [e for e in self.all_loss_nams if "loss_dice" not in e ]
        self.all_loss_nams = [e for e in self.all_loss_nams if "loss_mask" not in e ]
    

        for name, p in self.glee.named_parameters():
            if p.requires_grad == True:
                print (" **** " , name, p.data.shape)
        

    def forward(self, batched_inputs, task='flicker'):
        if self.training:
            _, prompt_list = self.get_task_name(batched_inputs)
        else:
            task, prompt_list = self.get_task_name(batched_inputs)
        batch_name_list = None

        if self.training:
            images = self.preprocess_image(batched_inputs, task)
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            
            targets, prompt_list = self.prepare_targets(batched_inputs, gt_instances, images, prompt_list, task)

            batch_name_list = self.dataset_name_dicts[task]  # usa all category list

            (outputs, mask_dict), track_loss, dist_loss  = self.glee(images, prompt_list, task, targets, batch_name_list)

            losses = self.criterion(outputs, targets, mask_dict, task)
            losses.update({"track_loss":track_loss})
            losses.update({"dist_loss":dist_loss})
            
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                
                if ('box' in k or 'giou' in k) and task == 'grit':
                    losses[k] *= 0
            this_loss_names = set(list(losses.keys()))
            for loss_name in self.all_loss_nams:
                assert loss_name in this_loss_names , "{} is not in this batch, task is {}".format(loss_name,task)

            return losses
        else:  # evaluation
            
            images = self.preprocess_image(batched_inputs, task)
            batch_name_list = self.dataset_name_dicts[task]

            (outputs,_),_,_ = self.glee(images, prompt_list, task, batch_name_list=batch_name_list, is_train=False)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            del outputs
            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]
                if True:
                    if self.is_lsj:
                        resize_ratio = image_size[0]/max(height, width)
                        crop_size =  (int(height*resize_ratio), int(width*resize_ratio))
                    else:
                        crop_size = image_size
                    # mask_pred_result = sem_seg_postprocess(
                    #     mask_pred_result, crop_size, height, width
                    # )
                    mask_pred_result = mask_pred_result[None,][:,:,:crop_size[0],:crop_size[1]]
                    mask_pred_result = F.interpolate( mask_pred_result,   size=(height,width),  mode="bilinear",  align_corners=False,      )[0]
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                # instance segmentation inference
                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    # height = new_size[0]/crop_size[0]*height
                    # width = new_size[1]/crop_size[1]*width
                    if self.is_lsj:
                        mask_box_result = self.LSJ_box_postprocess(mask_box_result, new_size, crop_size, height, width)
                    else:
                        height = new_size[0]/crop_size[0]*height
                        width = new_size[1]/crop_size[1]*width
                        mask_box_result = self.box_postprocess(mask_box_result, height, width)
                    instance_r = self.instance_inference(mask_cls_result, mask_pred_result, mask_box_result, task)
                    processed_results[-1]["instances"] = instance_r
            return processed_results

    def prepare_targets(self, batched_inputs, targets, images, prompt_list, task):
        img_long_size = max(images.tensor.shape[-2:])  # video data set into prompt mode with a probability of 0.4
        
        prompt_flag = False
        
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        for targets_per_image in targets:
            padded_masks = None

            gt_classes = targets_per_image.gt_classes
            
            image_size_xyxy = torch.as_tensor([w_pad, h_pad, w_pad, h_pad], dtype=torch.float, device=self.device)
            gt_boxes = box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
            gt_boxes = torch.clamp(gt_boxes,0,1)

            if prompt_flag and len(gt_classes)>0:
                # num_prompts = random.randint(1,len(gt_classes))
                num_prompts = 1 
                sample_ids = random.sample(list(range(0, len(gt_classes))), num_prompts)
                if padded_masks is not None:
                    padded_masks = padded_masks[sample_ids]
                gt_classes = gt_classes[sample_ids]
                gt_boxes = gt_boxes[sample_ids]
            else:
                if prompt_flag:
                    prompt_flag = False
                    prompt_list.pop("spatial")
                
                
            new_targets.append(
                {
                    "labels": gt_classes,
                    "masks": padded_masks,
                    "boxes":gt_boxes,
                }
            )
            if prompt_flag:
                prompt_list["spatial"].append(padded_masks)
        return new_targets, prompt_list

    







@META_ARCH_REGISTRY.register()
class GLEE_DUMP(GLEE):
    def __init__(self, cfg):
        super().__init__(cfg)

        matcher = self.glee.matcher

        del self.glee
        self.glee = GLEE_Model_DUMP(cfg, matcher, self.device, self.video_info, cfg.MODEL.CONTRAS_MEAN)


    def forward(self, batched_inputs):
        assert self.training is False 

        task, prompt_list = self.get_task_name(batched_inputs)
        batch_name_list = None

        images = self.preprocess_image(batched_inputs, task)
        batch_name_list = self.dataset_name_dicts[task]

        BACKBONE, NECK, ENCODED = self.glee(images, prompt_list, task, batch_name_list=batch_name_list, is_train=False)

        # print([e.shape for e  in BACKBONE], NECK.shape, ENCODED.shape)
        
        return BACKBONE, NECK, ENCODED
        
        


