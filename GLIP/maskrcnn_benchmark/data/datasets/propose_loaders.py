import os 
import sys
import math 

from functools import partial
import pandas as pd 
import numpy as np 
import random
import json
from PIL import Image, ImageDraw
import albumentations as A

import torch
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.transforms import transforms
# from torchvision.utils import save_image



from einops import rearrange, repeat
from maskrcnn_benchmark.config import cfg
from .coco  import COCODataset, BoxList, SegmentationMask




from .perturb_util import low_resolution2, apply_perturb, apply_atmospheric_perturb, \
    normalize, do_nothing, save_image, apply_multiple_noises, low_resolution_MULTI, low_resolution_preprocess, display_box_over_img


try:
    from mmcv.runner.utils import set_random_seed
except:
    from mmengine.runner import set_random_seed


def pertubation_methods(mode, train_mode=False ):
    post_perturb = do_nothing
    perturb = do_nothing
    if mode == "pixel_dropout":
        if train_mode:
            perturb_fn = A.PixelDropout (dropout_prob=0.1, per_channel=False, drop_value=0, mask_drop_value=None, p=1)
        else:
            perturb_fn = A.Compose([A.PixelDropout (dropout_prob=0.1, per_channel=False, drop_value=0, mask_drop_value=None, p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "salt_pepper":
        if train_mode:
            perturb_fn = A.SaltAndPepper (amount=(0.06, 0.1), salt_vs_pepper=(0.4, 0.6), p=1)
        else:
            perturb_fn = A.Compose([A.SaltAndPepper (amount=(0.06, 0.1), salt_vs_pepper=(0.4, 0.6), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "iso_blur":
        if train_mode:
            perturb_fn = A.ISONoise(color_shift=(0.08, 0.5), intensity=(0.4, 1), p=1)
        else:
            perturb_fn = A.Compose([A.ISONoise(color_shift=(0.08, 0.5), intensity=(0.4, 1), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    

    elif mode == "low-res" or mode == "low-res_MULTI":
        if mode == "low-res_MULTI":
            post_perturb = low_resolution_MULTI
        else:
            post_perturb = low_resolution2
    elif mode == "low-res2":
        perturb = low_resolution_preprocess
    elif mode == "low-res3":
        perturb = g = partial(low_resolution_preprocess, sev=2 ** 5)
    
    elif mode == "focus_blur":
        if train_mode:
            perturb_fn = A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=1)
        else:
            perturb_fn = A.Compose([A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "chromatic":
        if train_mode:
            perturb_fn = A.ChromaticAberration(primary_distortion_limit=(-1, 1), secondary_distortion_limit=(-2, 2), mode='green_purple', interpolation=1, p=1)
        else:
            perturb_fn = A.Compose([A.ChromaticAberration(primary_distortion_limit=(-1, 1), secondary_distortion_limit=(-2, 2), mode='green_purple', interpolation=1, p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    
    elif mode == "jpg_compression":
        if train_mode:
            perturb_fn = A.ImageCompression (compression_type='jpeg', quality_range=(2, 30), p=1)
        else:
            perturb_fn = A.Compose([A.ImageCompression (compression_type='jpeg', quality_range=(10, 20), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "motion_blur2" or mode == "motion_blur2_MULTI" or mode == "Pickable_Motion_Blur_Generator":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        parent_dir = os.path.dirname(root)
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))

        from Weather_Simulation.motionblur_effect import Motion_Blur_Generator, Pickable_Motion_Blur_Generator
        if mode == "Pickable_Motion_Blur_Generator":
            perturb = Pickable_Motion_Blur_Generator()
        else:
            perturb = Motion_Blur_Generator()
    
    elif mode == "fog":
        if train_mode:
            perturb_fn = A.RandomFog(alpha_coef=0.1, fog_coef_range=(0.3, 1), p=1)
        else:
            perturb_fn = A.Compose([A.RandomFog(alpha_coef=0.1, fog_coef_range=(0.3, 1), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "rain_model2":
        from maskrcnn_benchmark.utils.dist import get_local_rank, get_rank
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        parent_dir = os.path.dirname(root)
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))
        from Weather_Simulation.weather import RainEffectGenerator2        
        # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device(f'cuda:{get_rank()}') if torch.cuda.is_available() else torch.device('cpu')
        perturb = RainEffectGenerator2(device=device)
        # post_perturb = RainEffectGenerator2()
    elif mode == "snow_model2":
        from maskrcnn_benchmark.utils.dist import get_local_rank, get_rank
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        parent_dir = os.path.dirname(root)
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))
        from Weather_Simulation.weather import SnowEffectGenerator2        
        device = torch.device(f'cuda:{get_rank()}') if torch.cuda.is_available() else torch.device('cpu')
        perturb = SnowEffectGenerator2(device=device)
    elif mode == "atmospheric" or mode == 'atmospheric_MULTI':
        from maskrcnn_benchmark.utils.dist import get_local_rank, get_rank
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ~/robustness_object_detection/GLIP/maskrcnn_benchmark/data/datasets/propose_loaders.py
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # ~/robustness_object_detection/GLIP/
        parent_dir = os.path.dirname(root)
        # ~/robustness_object_detection/
        sys.path.append(parent_dir)

        from TurbulenceSimulatorPython.turbStats import tilt_mat, corr_mat, get_r0
        from TurbulenceSimulatorPython.simulator import Simulator
        from TurbulenceSimulatorPython.helper import factorixze , normalize

        # Set turbulence parameters
        device = torch.device(f'cuda:{get_rank()}') if torch.cuda.is_available() else torch.device('cpu')
        size = 512
        N = 512  # Image size
        D = 0.1  # Aperture diameter
        r0 = 0.05  # Fried parameter
        L = 3000  # Propagation distance
        # Run tilt_mat function
        tilt_mat(N, D, r0, L, save_path=f'{parent_dir}/TurbulenceSimulatorPython/data')
        resize_transform1 = transforms.Resize( (size, size ) )

        correlation = -0.1 # [-0.1, -0.01, -1, -5]
        simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{parent_dir}/TurbulenceSimulatorPython/data', device=device).to(device, dtype=torch.float32)
        simulator.eval()

        post_perturb = apply_atmospheric_perturb(simulator=simulator, resize_transform1=resize_transform1, device=device)
    

    
    elif mode == "rain":
        if train_mode:
            perturb_fn = A.RandomRain(slant_range=(-20, 20), drop_length=2, drop_width=1, drop_color=(200, 200, 200), blur_value=10, brightness_coefficient=0.75, rain_type='heavy', p=1)
        else:
            perturb_fn = A.Compose([A.RandomRain(slant_range=(-20, 20), drop_length=2, drop_width=1, drop_color=(200, 200, 200), blur_value=10, brightness_coefficient=0.75, rain_type='heavy', p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "motion_blur":
        if train_mode:
            perturb_fn = A.MotionBlur(blur_limit=(5, 9), allow_shifted=True, angle_range=(0, 360), direction_range=(-1.0, 1.0), p=1)
        else:
            perturb_fn = A.Compose([A.MotionBlur(blur_limit=(5, 9), allow_shifted=True, angle_range=(0, 360), direction_range=(-1.0, 1.0), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "gaussian_blur":
        if train_mode:
            perturb_fn = A.GaussianBlur(blur_limit=0, sigma_limit=(2.0, 5.0), p=1)
        else:
            perturb_fn = A.Compose([A.GaussianBlur(blur_limit=0, sigma_limit=(2.0, 5.0), p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "snow":
        if train_mode:
            perturb_fn = A.RandomSnow (brightness_coeff=1, snow_point_range=(0.2, 0.4), method='texture', p=1)
        else:
            perturb_fn = A.Compose([A.RandomSnow (brightness_coeff=1, snow_point_range=(0.2, 0.4), method='texture', p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "mud":
        if train_mode:
            perturb_fn = A.Spatter (mean=(0.65, 0.65), std=(0.4, 0.4), gauss_sigma=(2, 2), cutout_threshold=(0.68, 0.68), intensity=(0.7, 0.7), mode='mud', color=None, p=1)
        else:
            perturb_fn = A.Compose([A.Spatter (mean=(0.65, 0.65), std=(0.4, 0.4), gauss_sigma=(2, 2), cutout_threshold=(0.68, 0.68), intensity=(0.7, 0.7), mode='mud', color=None, p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    elif mode == "grey":
        if train_mode:
            perturb_fn = A.ToGray (num_output_channels=3, method='weighted_average', p=1)
        else:
            perturb_fn = A.Compose([A.ToGray (num_output_channels=3, method='weighted_average', p=1)], seed=0)
        perturb = apply_perturb(perturb_fn)
    
    elif mode == "snow_model":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ~/robustness_object_detection/GLIP/maskrcnn_benchmark/data/datasets/propose_loaders.py
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # ~/robustness_object_detection/GLIP/
        parent_dir = os.path.dirname(root)
        # ~/robustness_object_detection/
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))

        from Weather_Simulation.weather import SnowEffectGenerator
        perturb = SnowEffectGenerator() 
    elif mode == "rain_model":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ~/robustness_object_detection/GLIP/maskrcnn_benchmark/data/datasets/propose_loaders.py
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # ~/robustness_object_detection/GLIP/
        parent_dir = os.path.dirname(root)
        # ~/robustness_object_detection/
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))

        from Weather_Simulation.weather import RainEffectGenerator
        perturb = RainEffectGenerator()
    
    elif mode == "fog_model":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # ~/robustness_object_detection/GLIP/maskrcnn_benchmark/data/datasets/propose_loaders.py
        root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        # ~/robustness_object_detection/GLIP/
        parent_dir = os.path.dirname(root)
        # ~/robustness_object_detection/
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))

        from Weather_Simulation.fog_effect import FogEffectGenerator
        perturb = FogEffectGenerator()
    elif "*" in mode:
        return do_nothing, do_nothing 
    elif mode == 'NONE':
        _ = 0 
    else:
        import pdb
        pdb.set_trace()
    return perturb, post_perturb           


class COCODataset_Perturb(COCODataset):
    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs)
        print(f"\n**** Testing Noise ::: {mode} **** \n")
        random.seed(0)
        np.random.seed(0)
        self.mode = mode        
        self.perturb, self.post_perturb = pertubation_methods(mode)
        self.__getitem__(0)
    
    def generic_steps(self, anno, img):
        # filter crowd annotations
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

        if self.max_box > 0 and len(boxes) > self.max_box:
            rand_idx = torch.randperm(self.max_box)
            boxes = boxes[rand_idx, :]
        else:
            rand_idx = None
        
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)

        if rand_idx is not None:
            classes = classes[rand_idx]
        if cfg.DATASETS.CLASS_AGNOSTIC:
            classes = torch.ones_like(classes)
        target.add_field("labels", classes)
        
        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "cbox" in anno[0]:
            cboxes = [obj["cbox"] for obj in anno]
            cboxes = torch.as_tensor(cboxes).reshape(-1, 4)  # guard against no boxes
            cboxes = BoxList(cboxes, img.size, mode="xywh").convert("xyxy")
            target.add_field("cbox", cboxes)

        if anno and "keypoints" in anno[0]:
            keypoints = []
            gt_keypoint = self.coco.cats[1]['keypoints']  # <TODO> a better way to get keypoint description
            use_keypoint = cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME
            for obj in anno:
                if len(use_keypoint) > 0:
                    kps = []
                    for name in use_keypoint:
                        kp_idx = slice(3 * gt_keypoint.index(name), 3 * gt_keypoint.index(name) + 3)
                        kps += obj["keypoints"][kp_idx]
                    keypoints.append(kps)
                else:
                    keypoints.append(obj["keypoints"])
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        return target

    def post_processing(self, target):
        if cfg.DATASETS.SAMPLE_RATIO != 0.0:
            ratio = cfg.DATASETS.SAMPLE_RATIO
            num_sample_target = math.ceil(len(target) * ratio) if ratio > 0 else math.ceil(-ratio)
            sample_idx = torch.randperm(len(target))[:num_sample_target]
            target = target[sample_idx]
        return target

    def __getitem__(self, idx):

        img, anno = super(COCODataset, self).__getitem__(idx)
        # img.save("temp_OG.png")

        target = self.generic_steps(anno, img)
        
        img = self.perturb(img)
        # img.save(f"temp-{self.mode}.png")
        # quit()
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        
        img = self.post_perturb(img)
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # quit()

        target = self.post_processing(target)        

        return img, target, idx

class COCODataset_Perturb_Debug(COCODataset_Perturb):
    
    def __getitem__(self, idx):

        img, anno = super(COCODataset, self).__getitem__(idx)
        # img.save("temp_OG.png")

        target = self.generic_steps(anno, img)
        
        img = self.perturb(img)
        # img.save(f"temp-{self.mode}.png")
        # quit()

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img = self.post_perturb(img)
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}_Newton.png" )
        # quit()

        target = self.post_processing(target)        

        return img, target, idx

    



from maskrcnn_benchmark.data.datasets.flickr import FlickrDataset
from maskrcnn_benchmark.data.datasets.modulated_coco import ConvertCocoPolysToMask, \
    has_valid_annotation, BoxList, sanity_check_target_after_processing
 

class FlickrDataset_MULTI_SCALE(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, return_tokens, is_train=False, 
        tokenizer=None, disable_clip_to_image=False, no_mask_for_gold=False, max_query_len=256, severity=3, mode=None, **kwargs):
        super(FlickrDataset_MULTI_SCALE, self).__init__(img_folder, ann_file)

        assert is_train == True, "Modified dataloader of Test? Something is wrong"
        self.setup(transforms, max_query_len, return_masks, return_tokens, tokenizer, is_train, disable_clip_to_image, no_mask_for_gold)
        idx = 0 
        self.mode = mode     

        self.perturb, self.post_perturb = pertubation_methods(mode, train_mode=True)
        # resize = self._transforms.transforms[0]
        # resize.min_size, resize.max_size,  resize.restrict
        self.__getitem__(idx)
        
    def setup(self, transforms, max_query_len, return_masks, return_tokens, tokenizer, is_train, disable_clip_to_image, no_mask_for_gold):
        self.ids = sorted(self.ids)
        ids = []
        for img_id in self.ids:
            if isinstance(img_id, str):
                ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
            else:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.max_query_len = max_query_len
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer, max_query_len=max_query_len)
        self.is_train = is_train
        self.disable_clip_to_image = disable_clip_to_image
        self.no_mask_for_gold = no_mask_for_gold

    def ann_info(self, idx, target):
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        anno = {"image_id": image_id, "annotations": target, "caption": caption}

        # This dataset is used for Flickr & Mixed, so the sequence is maskable
        anno["greenlight_span_for_masked_lm_objective"] = [(0, len(caption))]
        if self.no_mask_for_gold:
            anno["greenlight_span_for_masked_lm_objective"].append((-1, -1, -1))
        
        return anno, dataset_name, coco_img

    def process_ann(self, anno, img, target):
        # convert to BoxList (bboxes, labels)
        boxes = torch.as_tensor(anno["boxes"]).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")
        classes = anno["labels"]
        target.add_field("labels", classes)
        
        if not self.disable_clip_to_image:
            num_boxes = len(target.bbox)
            target = target.clip_to_image(remove_empty=True)
            assert num_boxes == len(target.bbox), "Box got removed in MixedDataset!!!"
        return target

    def add_attributes(self, anno, target, dataset_name, coco_img):
        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])
        target.add_field("dataset_name", dataset_name)
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target.add_field(extra_key, coco_img[extra_key])

    


    def display_box_over_img(self, img, anno, border_width = 5, border_color = 'red', mode="xywh", direct_box=False):
        img.save("temp.png")
        draw = ImageDraw.Draw(img)
        for e in anno:
            ## xmin , ymin , width, height
            if direct_box :
                rectangle_coords = e
            else:
                rectangle_coords = e['bbox']    
            if mode == "xywh":
                rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
            for i in range(border_width):
                draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
        img.save("temp2.png")

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
    
    def __getitem__(self, idx):
        img, target = super(FlickrDataset_MULTI_SCALE, self).__getitem__(idx)
        anno, dataset_name, coco_img = self.ann_info(idx, target)
        
        img, anno = self.prepare(img, anno)
        img_LQ = self.perturb(img)

        ########################################## VERIFY BOXES AND IMAGES ARE CORRECT 
        # self.display_box_over_img(img, anno['boxes'], mode="xywh", direct_box=True)
        # os.system("rsync -a ~/VLM-LR/glip/GLIP/*.png ucf2:~/VLM-LR/glip/GLIP/")        
        
        target = self.process_ann(anno, img, target)
        if self._transforms is not None:
            RANDOM_SEED = random.randint(0,2**32-1) 
            set_random_seed(RANDOM_SEED)
            img_LQ, _ = self._transforms(img_LQ, target)
            set_random_seed(RANDOM_SEED)
            img, target = self._transforms(img, target)


        self.add_attributes(anno, target, dataset_name, coco_img)
        sanity_check_target_after_processing(target)

        img_LQ  = self.post_perturb(img_LQ)
        img = [img, img_LQ]
        img = torch.stack(img)
        # save_image( normalize( img[:,torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        
        
        return img, target, idx


class FlickrDataset_MULTI_SCALE_MULTI_LR(FlickrDataset_MULTI_SCALE):
    def __init__(self, N_BR=2, mode=None, **kwargs):
        self.N_BR = N_BR - 1
        mode += "_MULTI"
        super().__init__(mode=mode,  **kwargs)

    def __getitem__(self, idx):
        img, target = super(FlickrDataset_MULTI_SCALE, self).__getitem__(idx)
        anno, dataset_name, coco_img = self.ann_info(idx, target)
        
        img, anno = self.prepare(img, anno)
        img_LQs = []
        for _  in range(self.N_BR):
            img_LQs.append( self.perturb(img) )
        
        target = self.process_ann(anno, img, target)
        if self._transforms is not None:
            RANDOM_SEED = random.randint(0,2**32-1) 
            for i  in range(self.N_BR):
                set_random_seed(RANDOM_SEED)
                img_LQs[i], _ = self._transforms(img_LQs[i], target)
            set_random_seed(RANDOM_SEED)
            img, target = self._transforms(img, target)


        self.add_attributes(anno, target, dataset_name, coco_img)
        sanity_check_target_after_processing(target)

        for i  in range(self.N_BR):
            img_LQs[i] = self.post_perturb(img_LQs[i])
        img = [img] + img_LQs
        img = torch.stack(img)
        # save_image( normalize( img[:,torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # quit()
        
        return img, target, idx



class FlickrDataset_MULTI_SCALE_LRCAPT(FlickrDataset_MULTI_SCALE):
    def __init__(self, img_folder=None, ann_file=None, N_LR=2, N_BR=2, **kwargs):
        
        import re, shutil
        
        assert "subset3" in ann_file or "debug" in ann_file, f"Different Subset of Captions not yet implemented... {ann_file}"
        variants_json = "../DATASET/subset_3_variants.json"
        with open(variants_json,'r') as f:
            variant_dict_data = json.load(f)
        variant_dict_data = variant_dict_data['variants']
        
        variant_dict = dict()
        pattern = r'\b(?:10|[1-9])\.\s*'
        repetative = 0 
        for e in variant_dict_data:
            file_name = e['file_name']
            data = e['text']
            data = data.replace("\n", " ").replace("  ", " ")
            data = re.split(pattern, data)
            data = [x for x in data if x != ""]
            # print(e['file_name'], data)
            assert len(data) == 10
            if file_name not in variant_dict: repetative+=1
            # assert file_name not in variant_dict, \
            #     (
            #         shutil.copyfile(os.path.join(img_folder, file_name), f'./{file_name}'),
            #         f"Repeated key {file_name}"
            #     )
                
            variant_dict[ file_name ] = data
        print(f"Repetative captions : {repetative}")
        self.variant_dict = variant_dict
        self.N_LR = N_LR
        self.N_BR = N_BR
        assert N_BR == 2, "Something is broken for N_BR != 2 in 'FlickrDataset_MULTI_SCALE_LRCAPT' "
        super().__init__(img_folder=img_folder, ann_file=ann_file, **kwargs)
     
    def __getitem__(self, idx):
        img, target, idx= super(FlickrDataset_MULTI_SCALE_LRCAPT, self).__getitem__(idx)

        # save_image( normalize( img[:,torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # target.extra_fields['caption']
        # display_box_over_img(transforms.ToPILImage()(normalize(img[0])), target.bbox, mode="xywh", direct_box=True)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]

        captions  = self.variant_dict[coco_img['file_name']]
        # for e in self.variant_dict[coco_img['file_name']]:e        
        
        captions = random.choices(captions, k=self.N_LR * self.N_BR)
        target.add_field("new_captions", captions)

        return img, target, idx


class FlickrDataset_MULTI_SCALE_LRCAPT_MULTI_AUG(FlickrDataset_MULTI_SCALE_LRCAPT):
    def __init__(self, mode=None, ONE_NOISE=None, **kwargs):
        self.Aug = apply_multiple_noises(ONE_NOISE=ONE_NOISE)
        if mode: 
            for noise in mode.split('*'):
                if ("atmospheric" in mode or "rain_model2" in mode) and noise  == 'motion_blur2':
                    noise = 'Pickable_Motion_Blur_Generator'
                perturb_local, post_perturb_local = pertubation_methods(noise, train_mode=True)
                self.Aug.add_noise(noise, perturb_local, post_perturb_local)
            self.Aug.setup()
        super().__init__(mode=mode, **kwargs)
        
    def __getitem__(self, idx):

        img, target = super(FlickrDataset_MULTI_SCALE, self).__getitem__(idx)
        anno, dataset_name, coco_img = self.ann_info(idx, target)
        img, anno = self.prepare(img, anno)

        img_LQ, noise = self.Aug.pre_method(img)        
        # img.save("temp.png"), img_LQ.save("temp2.png")
        
        target = self.process_ann(anno, img, target)
        if self._transforms is not None:
            RANDOM_SEED = random.randint(0,2**32-1) 
            set_random_seed(RANDOM_SEED)
            img_LQ, _ = self._transforms(img_LQ, target)
            set_random_seed(RANDOM_SEED)
            img, target = self._transforms(img, target)

        self.add_attributes(anno, target, dataset_name, coco_img)
        sanity_check_target_after_processing(target)

        # save_image( normalize( torch.stack([img, img_LQ])[:,torch.tensor((2, 1, 0))] ), f"temp-{self.mode}-pre.png" )
        img_LQ = self.Aug.post_method(img_LQ,  noise )        
        img = [img, img_LQ]
        img = torch.stack(img)
        # save_image( normalize( img[:,torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # quit()
        
        ###### LR CAPTIONS 
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]

        captions  = self.variant_dict[coco_img['file_name']]
        captions = random.choices(captions, k=self.N_LR * self.N_BR)
        target.add_field("new_captions", captions)

        return img, target, idx

        

        


class FlickrDataset_MULTI_SCALE_SEED(FlickrDataset_MULTI_SCALE_LRCAPT_MULTI_AUG):
    def __init__(self, N_LR=2, N_BR=2, **kwargs):
        super().__init__(**kwargs)
        self.N_LR = N_LR
        self.N_BR = N_BR
        self.Aug.LABEL_NOISE = True 
        # ['pixel_dropout', 'iso_blur', 'salt_pepper', 'low-res', 'focus_blur', 'jpg_compression', 
        # 'Pickable_Motion_Blur_Generator', 'fog', 'rain_model2', 'snow_model2', 'atmospheric'])
        if 'Pickable_Motion_Blur_Generator' in self.Aug.list_noises():
            self.Aug.noises['motion_blur2'] = self.Aug.noises['Pickable_Motion_Blur_Generator']
            
    def extra_captions(self, idx):
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        captions  = self.variant_dict[coco_img['file_name']]
        captions = random.choices(captions, k=self.N_LR * self.N_BR)
        return captions

    def __getitem__(self, idx, extra_cations=True):
        img, target = super(FlickrDataset_MULTI_SCALE, self).__getitem__(idx)
        anno, dataset_name, coco_img = self.ann_info(idx, target)
        img, anno = self.prepare(img, anno)

        target = self.process_ann(anno, img, target)
        if self._transforms is not None:
            RANDOM_SEED = random.randint(0,2**32-1) 
            set_random_seed(RANDOM_SEED)
            img, target = self._transforms(img, target)

        self.add_attributes(anno, target, dataset_name, coco_img)
        sanity_check_target_after_processing(target)

        ###### LR CAPTIONS 
        if extra_cations:
            captions = self.extra_captions( idx)
            target.add_field("new_captions", captions)

        return img.unsqueeze(0), target, idx, RANDOM_SEED

    def __call__(self, idx, seed= 2, noise=None):
        
        img, target = super(FlickrDataset_MULTI_SCALE, self).__getitem__(idx)
        anno, _, _ = self.ann_info(idx, target)
        img, anno = self.prepare(img, anno)

        # self.Aug.noises.keys()
        
        img_LQ, noise = self.Aug.pre_method(img, noise)        
        
        target = self.process_ann(anno, img, target)
        if self._transforms is not None:
            set_random_seed(seed)
            img_LQ, _ = self._transforms(img_LQ, target)
            
        img_LQ = self.Aug.post_method(img_LQ,  noise)        
        # save_image( normalize( img_LQ[torch.tensor((2, 1, 0))] ), f"temp-{noise}.png" )
        return img_LQ







class Load_Feats(data.Dataset):
    def __init__(self, cfg=None, ext_dataset=None, ext_csv = None, ext_csv_mode=None, load_feats=True, debug=None, **kwargs):
        super().__init__()
        self.ext_dataset = None 
        if ext_dataset:
            self.ext_dataset = True 
            self.ext_df = self.load_exp_ext_pattern(ext_csv, ext_csv_mode)

        self.N_candidates = cfg.DATALOADER.N_CANDIDATES
        # df = self.load_exp_pattern('../Analysis/GLIP-T_5-LR_TKO-summary.csv')
        self.df = self.load_exp_pattern(cfg.SOLVER.TARGET_CSV)
        self.eval = False 
        self.feat_path = cfg.SOLVER.FEAT_PATH

        self.index_to_noise = {v:k for v,k in enumerate(cfg.NOISES)}
        self.feats_lib = {}
        self.sample_size = cfg.MODEL.SAMPLE

        if load_feats:
            for noise in cfg.NOISES:
                feats = self.load_feats(self.feat_path, noise, H=cfg.MODEL.SIZE, factor=cfg.MODEL.FACTOR, )
                self.feats_lib[noise] = feats
                if debug:break 
        
        if ext_dataset:
            for noise in ext_dataset:
                feats = self.load_feats(self.feat_path, noise, H=cfg.MODEL.SIZE, factor=cfg.MODEL.FACTOR, prefix_path='ALL_BACKBONE')
                if noise == 'bdd100k_train':
                    noise = 'BBDK'
                elif noise == 'dawn_train':
                    noise = 'DAWN'
                elif noise == 'FoggyCityscape_amodal_train':
                    noise = 'FoggyCity-Amodal'
                elif noise == 'FoggyCityscape_modal_train':
                    noise = 'FoggyCity-Modal'

                if debug:break 
                self.feats_lib[noise] = feats
            self.eval = True 
            self.ext_dataset_mode('BBDK')
            self.ext_data_list = []

        self.no_of_noise = len(self.index_to_noise)
        N = self.__len__()
        self.__getitem__(random.randint(0, N))

    def load_exp_ext_pattern(self, file_name, ext_csv_mode):
        df = self.read_csv(file_name)
        df['model_type'] = df.apply(lambda x : x.name.split("-")[2].lower(), 1 )
        df['filter'] = df.apply(lambda x : 1 if ("trial" not in x.name.lower() and "chromatic" not in x.name.lower()) else 0, 1 )
        df = df[ (df['filter'] != 0) & (df['model_type'] != 'lr_tko_sr') ]
        df = df[df.model_type == ext_csv_mode]
        df['noise'] = df.apply(lambda x : x.name.split("-")[-1] if "low-res" not in x.name else "low-res", 1)
        df = df.drop(['filter', 'model_type'], axis=1)
        discarded = [e for e in df.columns if "WEDGE" in e]
        df = df.drop(discarded, axis=1)
        # df = df.set_index('noise').T
        return df 
        
    def load_feats(self, feat_path, noise, H, factor, prefix_path = 'ALL_BACKBONE_SPATIAL_coco'):
        x = torch.load( os.path.join(feat_path, f'{prefix_path}_{noise}.pth') )
        x = x['all_feats']
        x = rearrange(x, "B C (H W) -> B C H W", H=H,  W=H)
        x = F.interpolate( x, size=(H // factor , H // factor), mode="bilinear", align_corners=False)
        x = rearrange(x, "B C H W -> B (H W) C")
        return x 

    def read_csv(self, file_name, subtract_rows = True):
        df = pd.read_csv(file_name, index_col=0)
        df = df.dropna()
        if subtract_rows:
            df = df.sub(df.iloc[0])
        df = df.iloc[1:]
        return df 

    def load_exp_pattern(self, file_name='', exp_result = { 'LR_TKO': None, 'Adapter': None, 'LoRA':None, 'VPT': None, }):
        # Train on rows  evalaute on columns
        df = self.read_csv(file_name)
        df = df.drop('OG', axis=1)
        # Train on columns  evalaute on rows
        df = df.T
        df = df.drop('chromatic', axis=1)
        df = df.drop('chromatic').drop('BBDK').drop('low-res2')        
        return df

    def __len__(self):
        if self.ext_dataset:
            return len(self.index_to_noise.keys())
        else:
            return len(self.index_to_noise.keys()) ** 2

    def sample_feats(self, name):
        x_name = self.index_to_noise[name]
        x = self.feats_lib[x_name]
        if self.eval:
            indices = torch.arange(1, x.shape[0], 20)
            indices=  indices[:100]
            x = rearrange( x[indices, ], "B N C -> B N C")
        else:
            indices = torch.randint(0, x.shape[0], (self.sample_size,) )
            x = rearrange( x[indices, ], "B N C -> 1 B N C")
        return x, x_name

    def ext_dataset_mode(self, dataset, drop_overall=None ):
        self.external_dataset_name = dataset

        self.local_ext_df = self.ext_df
        self.local_ext_df = self.local_ext_df.set_index('noise')

        columns= [e for e in self.local_ext_df.columns if dataset in e]
        self.local_ext_df = self.local_ext_df[ columns ]
        if drop_overall: 
            columns = [e for e in self.local_ext_df if "Overall" not in e]
            self.local_ext_df = self.local_ext_df[ columns ]
        
        columns = {e: e.replace('BBDK-WEATHER-', '').replace('WEDGE-', '').replace('DAWN-', '').replace('FoggyCity-Amodal-', '').replace('BBDK-', '').replace('FoggyCity-Modal-', '') for e in columns}
        self.local_ext_df = self.local_ext_df.rename(columns=columns)
        columns = list(self.local_ext_df.columns)
        self.local_ext_df = self.local_ext_df.T
        self.ext_data_list = columns


    def __getitem__(self, idx):
        if self.ext_dataset:
            anchor = idx 
            x_f, x_name = self.sample_feats(anchor)
            y_f = self.feats_lib[self.external_dataset_name]
            y_name = self.external_dataset_name
            labels = torch.tensor(self.local_ext_df[x_name])
            self.local_ext_df[x_name]            
            return x_f, y_f, labels, [x_name] +  list(self.local_ext_df[x_name].keys())
        else:
            anchor = idx // self.no_of_noise
            x_f, x_name = self.sample_feats(anchor)
            candidate = idx % self.no_of_noise
            y_f, y_name = self.sample_feats(candidate)
        
        if self.eval:
            labels = torch.tensor([self.df[x_name][y_name]])
            return x_f, y_f, labels, [x_name, y_name]
        else:
            rand_candidate = random.randint(0, self.no_of_noise - 1 )
            z_f, z_name = self.sample_feats(rand_candidate)
            c_f = torch.cat([y_f, z_f])
            labels = [self.df[x_name][y_name], self.df[x_name][z_name]]

            for i in range(self.N_candidates - 1):
                rand_candidate = random.randint(0, self.no_of_noise - 1 )
                z_f, z_name = self.sample_feats(rand_candidate)
                c_f = torch.cat([c_f, z_f])
                labels.append(self.df[x_name][z_name])
                
            labels = torch.tensor(labels)
            return x_f, c_f, labels
        
class Load_Feats_Categories(Load_Feats):
    def __init__(self, cfg=None, ext_dataset=None, ext_csv = None, ext_csv_mode=None, **kwargs):
        self.ext_noises = []
        super().__init__(cfg=cfg, **kwargs)
        self.ext_dataset = None 
        if ext_dataset:
            self.ext_dataset = True 
            self.ext_df = self.load_exp_ext_pattern(ext_csv, ext_csv_mode)

        feat_path = cfg.SOLVER.FEAT_PATH
        self.eval = True 
        self.ext_noises_noises = []
        if ext_dataset:
            for noise in ext_dataset:
                if 'bdd100k' in noise:
                    noise_name = 'BBDK'
                elif 'dawn' in noise:
                    noise_name = 'DAWN'
                feats = self.load_feats_w_categories(feat_path, noise, H=cfg.MODEL.SIZE, factor=cfg.MODEL.FACTOR, prefix_path='ALL_BACKBONE', noise_name=noise_name)
                self.feats_lib.update(feats)
                self.ext_noises_noises += list(feats.keys())

            self.ext_dataset_mode('BBDK')
            self.ext_data_list = []

        self.no_of_noise = len(self.index_to_noise)
        N = self.__len__()
        self.__getitem__(random.randint(0, N))

    def load_feats_w_categories(self, feat_path, noise, H, factor, prefix_path = 'ALL_BACKBONE_SPATIAL_coco', noise_name=None):
        x = torch.load( os.path.join(feat_path, f'{prefix_path}_{noise}.pth') )
        categories = x['categories']
        x = x['all_feats']
        x = rearrange(x, "B C (H W) -> B C H W", H=H,  W=H)
        x = F.interpolate( x, size=(H // factor , H // factor), mode="bilinear", align_corners=False)
        x = rearrange(x, "B C H W -> B (H W) C")

        categories = np.array(categories)
        feats = {}
        for e in set(categories):
            index = categories == e
            feats[noise_name + "_" + e ] = x[index]
            
        return feats

    def __len__(self):
        return len(self.index_to_noise.keys()) * len(self.ext_noises)

    def ext_dataset_mode(self, dataset):
        super().ext_dataset_mode(dataset=dataset, drop_overall=True )
        column = [e for e in self.ext_noises_noises if dataset in e]
        column = [e.replace(self.external_dataset_name, '')[1:] for e in column]
        self.local_ext_df = self.local_ext_df.T[column].T
        # self.ext_noises_noises
        # self.external_dataset_name
        # self.local_ext_df
        self.ext_noises = column
        # self.ext_data_list       
        

    def __getitem__(self, idx):
        # 77 --> 0 | 1| 2 .... 7  
        # self.__len__(), len(self.ext_noises)
        anchor = idx // self.no_of_noise
        candidate = idx % self.no_of_noise

        if self.ext_dataset:
            x_f, x_name = self.sample_feats(candidate)
            y_name = self.ext_noises[anchor] 
            y_f = self.feats_lib[ self.external_dataset_name + '_' + y_name ]
            
            labels = self.local_ext_df[x_name][y_name]
            labels = torch.tensor(labels)
            
            return x_f, y_f, labels, [x_name, y_name] 
        else:    
            x_f, x_name = self.sample_feats(anchor)
            y_f, y_name = self.sample_feats(candidate)
        
        if self.eval:
            labels = torch.tensor([self.df[x_name][y_name]])
            return x_f, y_f, labels, [x_name, y_name]
        


