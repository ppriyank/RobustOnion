import copy 
import torch 
import numpy as np 
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .refcoco_dataset_mapper import RefCOCODatasetMapper
from .proposed_utils import display_box_over_img

__all__ = ["External_Dataset"]


class External_Dataset(RefCOCODatasetMapper):
    
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg=cfg, is_train=is_train)
        assert is_train is False 

        meta_data = ['person']
        self.dataset_name = 'coco'
        self.process_path = False 
        if "FoggyCitiscape" in cfg.DATASETS.TEST[0]:
            self.level= cfg.TEST.FOG_LEVEL
            self.process_path = True 
            meta_data = ['car',  "bicycle", 'bus', 'motorcycle',  'truck']
            self.dataset_name = 'city'
        elif "wedge" in cfg.DATASETS.TEST[0]:
            meta_data = ['person',  "bicycle", 'car', "motorcycle", "van", 'bus', 'truck']
            self.dataset_name = 'wedge'
        elif "dawn" in cfg.DATASETS.TEST[0]:
            meta_data = ['person',  "bicycle", 'car',  "motorcycle",  'bus', 'truck']
            self.dataset_name = 'dawn'
        elif "bddk" in cfg.DATASETS.TEST[0]:
            meta_data = ['person', 'car',  "rider", 'bus', 'truck', "bike", 'motor', 'traffic light',  "traffic sign"]
            self.dataset_name = 'bddk'
        
        self.mapper = {i:e for i,e in enumerate(meta_data)}
        

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if self.process_path:
            filename = dataset_dict["file_name"]
            filename = filename + f'_leftImg8bit_foggy_beta_{self.level}.png'
            dataset_dict["file_name"] = filename            
        

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        # display_box_over_img(image, annos, mapper=self.mapper, mode="xyxy", direct_box=False)
        
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        dataset_dict["instances"],_mask = utils.filter_empty_instances(instances, return_mask=True)

        dataset_dict['dataset_name'] = self.dataset_name
        return dataset_dict
        

