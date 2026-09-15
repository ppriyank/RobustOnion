

from .coco  import COCODataset
from .modulated_coco import CocoGrounding, convert_od_to_grounding_simple, BoxList, sanity_check_target_after_processing

import copy 
import time
import random 
import torch
import numpy as np 
from PIL import Image, ImageDraw

from collections import defaultdict
from .perturb_util import display_box_over_img, save_image, normalize, set_seeds, get_random_seed
from maskrcnn_benchmark.engine.utils import save_img

# https://proceedings.neurips.cc/paper_files/paper/2024/file/89d0d5c2f720921df93bbb8fef514571-Paper-Conference.pdf
# https://proceedings.neurips.cc/paper_files/paper/2024/file/e1fcd183ab33714a8464e4e9a20ac710-Paper-Conference.pdf

class DEBUG_CLASS_Grounding(CocoGrounding):
    def __init__(self, ann_file=None, **kwargs):
        del kwargs['remove_images_without_annotations']
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 

        super().__init__(ann_file=ann_file, remove_images_without_annotations=remove_images_without_annotations, **kwargs)
    
    def demo_run(self, idx, selective_obj=None, enable_text=False):
        img_id = [self.ids[idx]]
        meta = self.coco.loadImgs(img_id)[0]
        path = meta['file_name']
        print("===", path)
        

        img, anno = super(CocoGrounding, self).__getitem__(idx)
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        kwargs = {}
        if enable_text:
            categories = self.categories(no_background=False)
            labels = [categories[obj["category_id"]] for obj in anno if obj["category_id"]]            
            kwargs['labels']= labels
            
        display_box_over_img(img, anno, border_width=2, **kwargs)
    

class DEBUG_CLASS(COCODataset):
    def __init__(self, ann_file=None, **kwargs):
        del kwargs['remove_images_without_annotations']
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 

        super().__init__(ann_file=ann_file, remove_images_without_annotations=remove_images_without_annotations, **kwargs)
     
    def demo_run(self, idx, selective_obj=None, enable_text=False, **args):
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        kwargs = {}
        if enable_text:
            categories = self.coco.dataset["categories"]
            category_id = {e['id']:e['name'] for e in categories}
            labels = [category_id[e['category_id']] for e in anno]
            kwargs['labels']= labels
        kwargs.update(args)
        display_box_over_img(img, anno, **kwargs)
    







class BDD100K(COCODataset):
    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        # self.demo_run(10)
        pass

    def demo_run(self, idx, selective_obj=None):
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        display_box_over_img(img, anno)


class BDD100K_Grounding(DEBUG_CLASS_Grounding):
    
    def __init__(self, img_folder, ann_file, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, return_masks=False, return_tokens=True, **kwargs):
        remove_images_without_annotations = False 
        if "train" in ann_file:
            remove_images_without_annotations = True 
            
        super().__init__(
            img_folder=img_folder, ann_file=ann_file,  remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, return_masks=return_masks, return_tokens=return_tokens, **kwargs
            )

        # all_classes = []
        # for i in range(self.__len__()):
        #     _, obj = super(CocoGrounding, self).__getitem__(i);
        #     classes = [e['category_id'] for e in obj]
        #     all_classes += classes
        # all_classes = np.array(all_classes)

        categories = self.categories(no_background=False)
        # for e in categories:e, (all_classes == e).sum()
        # (1, 91349)
        # (2, 713211)
        # (3, 4517)
        # (4, 11672)
        # (5, 29971)
        # (6, 7210)
        # (7, 3002)
        # (8, 186117)
        # (9, 239686)

        print(f"No of Data points : {len(self.ids)}")
        print(f"Classes : : {categories}")
        # self.demo_run(idx=100)
        # self.__getitem__(0)
    
        


class DAWN(COCODataset):
    # The DAWN dataset comprises a
    # collection of 1000 images from real-traffic environments, which
    # are divided into four sets of weather conditions: fog, snow, rain
    # and sandstorms

    # https://data.mendeley.com/datasets/766ygrbt8y/3 (V3)
    # https://www.mdpi.com/1424-8220/23/20/8471

    # 999 Total Images
    #  TOTAL NO OF IMAGES ::  1026
    #  TOTAL NO OF BOXES ::  7845
    
    # Weathers  : Fog 270
    # Weathers  : Rain 182
    # Weathers  : Snow 173
    # Weathers  : Sand 301

    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        
        for i in range(len(self.ids)):
            file_name = self.coco.loadImgs( self.ids[i] )[0]['file_name']
            img, anno = super(COCODataset, self).__getitem__(i)
                
        # self.json_category_id_to_contiguous_id
        # self.contiguous_category_id_to_json_id
        # self.coco.getCatIds()
        
        # self.demo_run(idx=0)
        # self.__getitem__(0)
    

    def demo_run(self, idx, selective_obj=None):
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        display_box_over_img(img, anno)
        


class WEDGE(COCODataset):
    
    # https://github.com/Infernolia/WEDGE/tree/main/Dataset/WEDGE
    # https://www.mdpi.com/1424-8220/23/20/8471

    # 999 Total Images
    #  TOTAL NO OF IMAGES ::  3353
    #  TOTAL NO OF BOXES ::  16236
    
    # Weathers  : cloudy 205
    # Weathers  : day 206
    # Weathers  : dust 204
    # Weathers  : fall 204
    # Weathers  : fog 197
    # Weathers  : hurricane 204
    # Weathers  : lightning 204
    # Weathers  : night 202
    # Weathers  : rain 202
    # Weathers  : snow 202
    # Weathers  : spring 203
    # Weathers  : summer 204
    # Weathers  : sun 206
    # Weathers  : tornado 208
    # Weathers  : windy 205
    # Weathers  : winter 204

    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        
        # self.demo_run(idx=500)
        # self.__getitem__(0)
    

    def demo_run(self, idx, selective_obj=None):
        img_id = self.ids[idx]
        img_id = [img_id]
        meta = self.coco.loadImgs(img_id)[0]
        path = meta['file_name']
        
        print("===", path)
        img, anno = super(COCODataset, self).__getitem__(idx)
        # img.save("temp.png")
        # filter crowd annotations

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)

        import pdb
        pdb.set_trace()
        
        categories = self.categories(no_background=False)
        labels = [categories[e['category_id']] for e in anno]
        
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        display_box_over_img(img, anno)
        


class WEDGE_Grounding(CocoGrounding):
    
    def __init__(self, img_folder, ann_file, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, return_masks=False, return_tokens=True, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            img_folder=img_folder, ann_file=ann_file,  remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, return_masks=return_masks, return_tokens=return_tokens, **kwargs
            )

        self.demo_run(idx=500)
        # import pdb
        # pdb.set_trace()
        self.__getitem__(0)
    

    def demo_run(self, idx, selective_obj=None):
        img_id = self.ids[idx]
        img_id = [img_id]
        meta = self.coco.loadImgs(img_id)[0]
        path = meta['file_name']
        print("===", path)
        img, anno = super(CocoGrounding, self).__getitem__(idx)
        
        # img.save("temp.png")
        # filter crowd annotations
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        categories = self.categories(no_background=False)
        labels = [categories[obj["category_id"]] for obj in anno if obj["category_id"]]
        display_box_over_img(img, anno, labels)
        


class FoggyCityscape(COCODataset):
    
    # train  val
    # train ::  TOTAL NO OF IMAGES ::  2908
    # val ::  TOTAL NO OF IMAGES ::  488

    # The values of the attenuation coefficient are 0.005, 0.01 and 0.02m-1 and correspond to visibility ranges of 600, 300 and 150m respectively. 
    # _foggy_beta_0.005.png #### LESS FOG 
    # _foggy_beta_0.01.png #### MEDIUM FOG 
    # _foggy_beta_0.02.png ##### DENSE FOG 

    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, fog_level=-1, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        
        def corse_level(level=0.02, prev_level=None):
            N = len(self.coco.imgs)
            for i in range(N):
                filename = self.coco.imgs[i]['file_name']
                if prev_level:
                    self.coco.imgs[i]['file_name'] = filename.replace(f'_foggy_beta_{prev_level}.png', f'_foggy_beta_{level}.png')
                else:
                    self.coco.imgs[i]['file_name'] = filename + f'_leftImg8bit_foggy_beta_{level}.png'
                
        if not ("train" in ann_file) and fog_level:
            corse_level(level=fog_level)

        # corse_level(level=0.02)
        # self.demo_run(idx=0)

        # corse_level(prev_level=0.02, level=0.01)
        # self.demo_run(idx=0)

        # corse_level(prev_level=0.01, level=0.005)
        # self.demo_run(idx=0)

        # corse_level(prev_level=0.005, level=0.02)
        # self.demo_run(idx=0)
        
        # self.__getitem__(0)
    

    def demo_run(self, idx, selective_obj=None):
        # print("===", self.coco.loadImgs(idx)[0]['file_name'])
        img, anno = super(COCODataset, self).__getitem__(idx)
        # filter crowd annotations
        if self.ignore_crowd:
            anno = [obj for obj in anno if obj["iscrowd"] == 0]
        if selective_obj:
            anno = [obj for obj in anno if obj["category_id"] == selective_obj]

        display_box_over_img(img, anno)
        



class VirtualKitti(DEBUG_CLASS):
    # https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):

        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs)

        ##### CHECK THIS SHOULD WORK AS EXPECTED .... if not coco annotation dumping is wrong 
        # self.demo_run(idx=0, enable_text=True)
        self.__getitem__(0)


class VirtualKitti_Grounding(DEBUG_CLASS_Grounding):
    # https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
    def __init__(self, img_folder, ann_file, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, return_masks=False, return_tokens=True,  mode=None, **kwargs):
        
        super().__init__(
            img_folder=img_folder, ann_file=ann_file, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, return_masks=return_masks, return_tokens=return_tokens, **kwargs
            )

        
        if mode:
            print("--", self.__len__())
            selected = []
            for key in self.ids:
                filename = self.coco.imgs[key]['file_name']
                if mode in filename:
                    selected.append(key)
                else:
                    del self.coco.imgs[key]
            # for key in selected:
            self.ids = selected
            print("--", self.__len__())
            self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        # use_caption_prompt || cfg.DATASETS.USE_CAPTION_PROMPT        
        #### CHECK THIS SHOULD WORK AS EXPECTED .... if not coco annotation dumping is wrong 
        # self.demo_run(idx=1, enable_text=True)
        self.__getitem__(1)





class VirtualKitti_Grounding_Pairs(VirtualKitti_Grounding):
    def __init__(self, N_BR=2, **kwargs):
        self.filename_io_idx = None 
        super().__init__(**kwargs)
        self.N_BR = N_BR - 2 
        condition_index = {
            'clone':0, 'fog':1, 'morning':2, 'overcast':3, 'rain':4, 'sunset':5, 
        }            
        # (, 2458)
        filename_io_idx = {}
        selected = []
        self.HD_label = 'clone'
        for key in self.ids:
            filename = self.coco.imgs[key]['file_name']
            if '15-deg' in filename or '30-deg' in filename:
                del self.coco.imgs[key]
                continue 
            else:
                selected.append(key)
            
            condition = filename.split('/')[1]
            filename = filename.replace(f"{condition}/frames/rgb/", "").replace("/", "_")
            index_position = condition_index[condition]
            if filename not in filename_io_idx:
                filename_io_idx[filename] = [0 for k in condition_index]
            filename_io_idx[filename][index_position] = key
        
        self.ids = selected
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.img_to_id_map = {v: k for k, v in enumerate(self.ids)}
        self.filename_io_idx = filename_io_idx
        N = self.__len__()
        
        # print("--", N)
        # for i in range(0, N, 50):
        #     self.__getitem__(i)
        #     time.sleep(0.1)
        self.__getitem__(1)
    
    def __getitem__(self, idx, return_pair=True ):
        if not self.filename_io_idx:return

        img, tgt = super(CocoGrounding, self).__getitem__(idx)
        # display_box_over_img(img, tgt, border_width=2)
        
        image_id = self.ids[idx]
        tgt = [obj for obj in tgt if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in tgt]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        classes = [obj["category_id"] for obj in tgt]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        target = target.clip_to_image(remove_empty=True)
        
        # Intended for COCO / ODinW
        annotations, caption, greenlight_span_for_masked_lm_objective = convert_od_to_grounding_simple(
            target=target,
            image_id=image_id,
            ind_to_class=self.ind_to_class,
            disable_shuffle=self.disable_shuffle,
            add_detection_prompt=self.add_detection_prompt,
            separation_tokens=self.separation_tokens,
            caption_prompt=self.caption_prompt if self.use_caption_prompt else None)
        
        anno = {"image_id": image_id, "annotations": annotations, "caption": caption}
        anno["greenlight_span_for_masked_lm_objective"] = greenlight_span_for_masked_lm_objective
        
        IMAGES = []
        img, anno = self.prepare(img, anno, box_format="xyxy")
        dummy_target = copy.deepcopy(target)
        local_seed = get_random_seed()
        set_seeds(local_seed)
        img, target = self._transforms(img, target)

        key = self.ids[idx]
        filename = self.coco.imgs[key]['file_name']
        condition = filename.split('/')[1]
        filename = filename.replace(f"{condition}/frames/rgb/", "").replace("/", "_")
        
        HQ_idx = self.filename_io_idx[filename][0]
        HQ_idx = self.img_to_id_map[HQ_idx]
        HQ_img, _ = super(CocoGrounding, self).__getitem__(HQ_idx)
        # display_box_over_img(HQ_img, tgt, border_width=2, name='demo_hq')
        set_seeds(local_seed)
        HQ_img, _ = self._transforms(HQ_img, dummy_target)
        IMAGES.append(HQ_img)

        IMAGES.append(img)
        for i in range(self.N_BR):
            LR_idx = random.choice(self.filename_io_idx[filename][1:])
            LR_idx = self.img_to_id_map[LR_idx]
            LQ_img, _ = super(CocoGrounding, self).__getitem__(LR_idx)
            # display_box_over_img(LQ_img, tgt, border_width=2, name='demo_LQ')
            set_seeds(local_seed)
            LQ_img, _ = self._transforms(LQ_img, dummy_target)
            IMAGES.append(LQ_img)
            
        img = torch.stack(IMAGES)
        # save_img(img, "temp_pair.png") 
        
        # add additional property
        for ann in anno:
            target.add_field(ann, anno[ann])
        
        sanity_check_target_after_processing(target)

        return img, target, idx

    

class WIDER_FACE(DEBUG_CLASS):

    # 3226 Total Images

    # blur_class = ["clear", "normal", "heavy"]
    # expression = ["typical", "exaggerate"]
    # illumination = ["normal", "exaggerate "]
    # occlusion = ["no", "partial", "heavy"]
    # pose = ["typical", "atypical"]


    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        
        # for i in range(len(self.ids)):
        #     file_name = self.coco.loadImgs( self.ids[i] )[0]['file_name']
        #     img, anno = super(COCODataset, self).__getitem__(i)
            # self.coco.imgs[original_id][weather_label]
                
        # self.coco.getCatIds()
        # self.demo_run(idx=0)
        # self.__getitem__(0)
    


class VIS_DRONE2019(DEBUG_CLASS):
    # 548 Total Images
    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        
        # self.demo_run(idx=0, enable_text=True, border_width=1)
        

class UAVDT(DEBUG_CLASS):
    # 548 Total Images
    def __init__(self, ann_file, root, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, **kwargs):
        
        if "train" in ann_file:
            remove_images_without_annotations = True 
        else:
            remove_images_without_annotations = False 
        
        super().__init__(
            ann_file=ann_file, root=root, remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, **kwargs
            )
        
        # self.demo_run(idx=0, enable_text=True, border_width=1)

class UAVDT_Grounding(DEBUG_CLASS_Grounding):
    
    def __init__(self, img_folder, ann_file, remove_images_without_annotations=None, transforms=None, ignore_crowd=True,
                 max_box=-1, few_shot=0, one_hot=False, override_category=None, return_masks=False, return_tokens=True, **kwargs):
        remove_images_without_annotations = False 
        if "train" in ann_file:
            remove_images_without_annotations = True 
            
        super().__init__(
            img_folder=img_folder, ann_file=ann_file,  remove_images_without_annotations=remove_images_without_annotations, 
            transforms=transforms, ignore_crowd=ignore_crowd, max_box=max_box, few_shot=few_shot, one_hot=one_hot, 
            override_category=override_category, return_masks=return_masks, return_tokens=return_tokens, **kwargs
            )

        categories = self.categories(no_background=False)
        print(f"No of Data points : {len(self.ids)}")
        print(f"Classes : : {categories}")
        # self.demo_run(idx=100)
        # self.__getitem__(0)
    

