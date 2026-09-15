import os 
import copy
import numpy as np
import torch
import re
from detectron2.data import detection_utils as utils

from detectron2.data import transforms as T

from detectron2.utils.file_io import PathManager
from .refcoco_dataset_mapper import RefCOCODatasetMapper
from PIL import Image

from .proposed_utils import do_nothing, pertubation_methods, normalize, save_image

from .joint_image_dataset_LSJ_mapper import Joint_Image_LSJDatasetMapper
from .joint_image_video_dataset_LSJ_mapper import Joint_Image_Video_LSJDatasetMapper

from detectron2.structures import BoxMode

__all__ = ["RefCOCODatasetMapper_Perturb"]


from detectron2.data.dataset_mapper import DatasetMapper

##### LVIS eval 
class DatasetMapper_Perturb(DatasetMapper):
    
    def set_up_mode(self, mode, sev=3):
        assert self.is_train is False 
        self.perturb = do_nothing
        self.perturb, self.post_perturb = pertubation_methods(mode, train_mode=False , sev=sev)
        self.mode = mode
        self.sev = sev 

    def other_setups(self, dataset_dict, image):
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
                
    def read_image(self, file_name, format):
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            
            image = utils._apply_exif_orientation(image)
            # image.save("temp_OG.png")    
            
            image = image.convert(format)
            image = self.perturb(image)
            # image.save(f"temp-{self.mode}.png")
            return utils.convert_PIL_to_numpy(image, format)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        
        image = self.read_image(file_name=dataset_dict["file_name"], format=self.image_format)
        dataset_dict = self.other_setups(dataset_dict, image)
        

        image = dataset_dict['image']
        # print(image.shape)
        image = self.post_perturb(image)
        # save_image( normalize( image ), f"temp-{self.mode}-{self.sev}.png" )
        # quit()
        
        dataset_dict['image'] = image


        return dataset_dict

##### COCO & ODinW-13
class RefCOCODatasetMapper_Perturb (RefCOCODatasetMapper):
    
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg=cfg, is_train=is_train)

        assert is_train is False 
        self.perturb = do_nothing
        self.perturb, self.post_perturb = pertubation_methods(cfg.DATASETS.DATALOADER_MODE, train_mode=False )
        self.mode = cfg.DATASETS.DATALOADER_MODE

    def other_setups(self, dataset_dict, image):
        if 'expressions' in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
            
            disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False
            dataset_dict["image"], image_shape, transforms = self.transform_img(image, disable_crop=disable_crop)
            if "expressions" in dataset_dict and dataset_dict["task"] == "grounding":
                dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                # language-guided detection
                task = dataset_dict["task"] if "task" in dataset_dict else None
                if self.lang_guide_det and task == "detection":
                    dataset_dict["expressions"] = self.prompt_test_dict[dataset_dict["dataset_name"]]
                    dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_dict["dataset_name"]]
                return dataset_dict

            if "annotations" in dataset_dict:
                instances, expressions_new = self.transform_annos(dataset_dict["annotations"], transforms, image_shape, dataset_dict)
                # add "expressions" for detection data
                dataset_dict["expressions"] = expressions_new
                instances = utils.filter_empty_instances(instances)

                if len(instances) == 0:
                    return None 
                dataset_dict["instances"] = instances
            if dataset_dict["task"] == "phrase_grounding":
                dataset_dict["task"] = "detection"
            return dataset_dict
        else:  # detection 
            if self.crop_gen is None:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                if np.random.rand() > 0.5:
                    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                else:
                    image, transforms = T.apply_transform_gens(
                        self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                    )

            image_shape = image.shape[:2]  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if False:#not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict
            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.mask_on:
                        anno.pop("segmentation", None)
                    anno.pop("keypoints", None)
                if 'task' in dataset_dict and dataset_dict['task']=='vg':
                    object_description_list = [ anno['object_description'] for anno in  dataset_dict["annotations"]]
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                dataset_dict["instances"],_mask = utils.filter_empty_instances(instances, return_mask=True)

                if 'task' in dataset_dict and dataset_dict['task']=='vg': # filter empty description
                    dataset_dict["object_descriptions"] = []
                    _mask = _mask.tolist()
                    assert len(_mask) == len(object_description_list)
                    for description, _m in zip(object_description_list,_mask):
                        if _m:
                            dataset_dict["object_descriptions"].append(description)
            return dataset_dict

    def load_image(self, dataset_dict):
        file_name =  dataset_dict["file_name"]

        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            image = utils._apply_exif_orientation(image)
            # image.save("temp_OG.png")    
            # image = utils._apply_exif_orientation(image)
            image = image.convert(self.img_format)
            image = self.perturb(image)
            # image.save(f"temp-{self.mode}.png")

            return utils.convert_PIL_to_numpy(image, self.img_format)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = self.load_image(dataset_dict)
        dataset_dict = self.other_setups(dataset_dict, image)

        image = dataset_dict['image']
        # print(image.shape)
        image = self.post_perturb(image)
        # save_image( normalize( image ), f"temp-{self.mode}.png" )
        dataset_dict['image'] = image

        # quit()
        return dataset_dict
        
####### EVA eval 
class Joint_Image_LSJDatasetMapper_Perturb(Joint_Image_LSJDatasetMapper):
    
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg=cfg, is_train=is_train)
        
        assert is_train is False 
        self.perturb = do_nothing
        if cfg.SEV != None:
            self.sev = cfg.SEV
        else:
            self.sev = 3 
        self.perturb, self.post_perturb = pertubation_methods(cfg.DATASETS.DATALOADER_MODE, train_mode=False , sev=self.sev )
        self.mode = cfg.DATASETS.DATALOADER_MODE

        # x = {
        #     'file_name': '/data/priyank/synthetic/coco/val2017/000000000285.jpg', 'height': 640, 'width': 586, 'image_id': 285, 
        #     'annotations': [
        #         {'iscrowd': 0, 'bbox': [1.43, 68.81, 584.57, 563.94], 'category_id': 21, 'segmentation': [[37.31, 373.02, 57.4, 216.61, 67.44, 159.21, 77.49, 113.29, 91.84, 86.03, 123.41, 84.59, 162.15, 96.07, 215.25, 86.03, 261.17, 70.24, 285.56, 68.81, 337.22, 68.81, 411.84, 93.2, 454.89, 107.55, 496.5, 255.35, 513.72, 262.53, 552.47, 292.66, 586.0, 324.23, 586.0, 381.63, 586.0, 449.08, 586.0, 453.38, 578.3, 616.97, 518.03, 621.27, 444.84, 624.14, 340.09, 625.58, 136.32, 625.58, 1.43, 632.75, 7.17, 555.26, 5.74, 414.64]], 
        #         'bbox_mode': None,
        #         }
        #     ], 
        #     'task': 'detection', 'dataset_name': 'coco'
        # }        
        # self.__call__(x)

    def other_setups(self, dataset_dict, image):
        utils.check_image_size(dataset_dict, image)
        if dataset_dict.get('task') == 'sa1b': # read the sa1b mask annotation which saved with images rather in annotation json
            mask_anno_json = json.load(open(dataset_dict["file_name"][:-3]+'json','rb'))
            assert len(mask_anno_json['annotations']) == len(dataset_dict['annotations'])
            for mask_anno, per_dict in zip(mask_anno_json['annotations'], dataset_dict['annotations']):
                per_dict['segmentation'] = mask_anno.get("segmentation", None)
        

        
        if 'expressions' in dataset_dict:  # refcoco data
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            
            disable_crop = self.has_ordinal_num(dataset_dict["expressions"]) if "expressions" in dataset_dict else False
            padding_mask = np.ones(image.shape[:2])
            if disable_crop:
                image, transforms = T.apply_transform_gens(self.tfm_gens_nocrop, image)
            else:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            image_shape = image.shape[:2]  # h, w
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if "expressions" in dataset_dict and dataset_dict["task"] == "grounding":
                dataset_dict["expressions"] = self.transform_expressions(dataset_dict["expressions"], transforms)
            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                # language-guided detection
                task = dataset_dict["task"] if "task" in dataset_dict else None
                if self.lang_guide_det and task == "detection":
                    dataset_dict["expressions"] = self.prompt_test_dict[dataset_dict["dataset_name"]]
                    dataset_dict["positive_map_label_to_token"] = self.positive_map_label_to_token_dict[dataset_dict["dataset_name"]]
                return dataset_dict

            if "annotations" in dataset_dict:
                # instances, expressions_new = self.transform_annos(dataset_dict["annotations"], transforms, image_shape, dataset_dict)
                # add "expressions" for detection data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict["annotations"]
                    if obj.get("iscrowd", 0) == 0
                ]

                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # Need to filter empty instances first (due to augmentation)
                instances = utils.filter_empty_instances(instances)
                h, w = instances.image_size
                # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
                # if hasattr(instances, 'gt_masks'):
                #     gt_masks = instances.gt_masks
                #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                #     instances.gt_masks = gt_masks
                if len(instances) == 0:
                    return None 
                dataset_dict["instances"] = instances
            # if dataset_dict["task"] == "phrase_grounding":
            #     dataset_dict["task"] = "detection"
            return dataset_dict
        else:  # detection dataset [coco obj365 UVO eta]
            padding_mask = np.ones(image.shape[:2])
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            # the crop transformation has default padding value 0 for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            image_shape = image.shape[:2]  # h, w
            W_wop = image_shape[1] - np.sum(padding_mask[0, :])
            H_wop = image_shape[0] - np.sum(padding_mask[:, 0])
            image_shape_wop = (H_wop, W_wop) # without padding
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

            if not self.is_train:
                # USER: Modify this if you want to keep them for some reason.
                dataset_dict.pop("annotations", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    # Let's always keep mask
                    anno.pop("keypoints", None)
                if dataset_dict.get('task') == 'vg' or dataset_dict.get('task') =='grit':
                    object_description_list = [ anno['object_description'] for anno in  dataset_dict["annotations"]]
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]

                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                # instances = utils.annotations_to_instances(annos, image_shape)
                instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                if 'gt_masks' in instances._fields.keys():
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                # Need to filter empty instances first (due to augmentation)
                # instances = utils.filter_empty_instances(instances)
                instances,_mask = utils.filter_empty_instances(instances, return_mask=True)
                h, w = instances.image_size
                # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
                # if hasattr(instances, 'gt_masks'):
                #     gt_masks = instances.gt_masks
                #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                #     instances.gt_masks = gt_masks
                # NOTE: Here we get the size of image without padding. 
                # This is different from the original Mask2Former
                setattr(instances, "_image_size", image_shape_wop)
                dataset_dict["instances"] = instances

                if dataset_dict.get('task') == 'vg' or dataset_dict.get('task') =='grit': # filter empty description
                    dataset_dict["object_descriptions"] = []
                    _mask = _mask.tolist()
                    assert len(_mask) == len(object_description_list)
                    for description, _m in zip(object_description_list,_mask):
                        if _m:
                            dataset_dict["object_descriptions"].append(description)
            return dataset_dict

    def read_image(self, file_name, format):
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            image = utils._apply_exif_orientation(image)
            # image.save("temp_OG.png")    
            # image = utils._apply_exif_orientation(image)
            image = image.convert(self.img_format)
            image = self.perturb(image)
            # image.save(f"temp-{self.mode}.png")
            return utils.convert_PIL_to_numpy(image, self.img_format)

    def __call__(self, dataset_dict):
        # print(dataset_dict)
        
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = self.read_image(dataset_dict["file_name"], format=self.img_format)
        
        dataset_dict = self.other_setups(dataset_dict, image)
        
        
        image = dataset_dict['image']
        # print(image.shape)
        image = self.post_perturb(image)
        # save_image( normalize( image ), f"temp-{self.mode}-{self.sev}.png" )
        # quit()
        dataset_dict['image'] = image

        return dataset_dict


####### Flicker Train 
class Joint_Image_Video_LSJDatasetMapper_Perturb(Joint_Image_Video_LSJDatasetMapper):
    def __init__(self, cfg, is_train=True, root=None):
        super().__init__(cfg=cfg, is_train=is_train)
        x = {
            'file_name': '/data/priyank/synthetic/flickr_dataset_30k/flickr30k/flickr30k-images/3359636318.jpg', 
            'height': 334, 'width': 500, 'expressions': 'Two people are talking outside of the video game shop next door to the mobile phone store .', 'image_id': 0, 
            'annotations': [
                {'iscrowd': 0, 'bbox': [144.0, 166.0, 64.0, 168.0], 'category_id': 0, 'tokens_positive': [[0, 10]], 'bbox_mode': BoxMode.XYWH_ABS},
                {'iscrowd': 0, 'bbox': [192.0, 1.0, 307.0, 230.0], 'category_id': 0, 'tokens_positive': [[67, 89]], 'bbox_mode': BoxMode.XYWH_ABS}, 
                {'iscrowd': 0, 'bbox': [1.0, 55.0, 168.0, 253.0], 'category_id': 0, 'tokens_positive': [[34, 53]], 'bbox_mode': BoxMode.XYWH_ABS}, 
                {'iscrowd': 0, 'bbox': [47.0, 183.0, 59.0, 151.0], 'category_id': 0, 'tokens_positive': [[0, 10]], 'bbox_mode': BoxMode.XYWH_ABS}
            ], 
            'task': 'phrase_grounding'
        }
        assert self.is_train == True 

        self.perturb = do_nothing
        self.perturb, self.post_perturb = pertubation_methods(cfg.DATASETS.DATALOADER_MODE, train_mode=True)
        self.mode = cfg.DATASETS.DATALOADER_MODE

        # self.lang_guide_det  
        self.ordinal_nums = ["first"]
        self.root = root
        self.__call__(x)

    def __call__(self, dataset_dict):
        return self.image_call(dataset_dict)
    
    def load_image(self, dataset_dict):
    
        file_name =  dataset_dict["file_name"]
        file_name = os.path.join(self.root, "flickr_dataset_30k", file_name.split("flickr_dataset_30k/")[1])
    
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        with PathManager.open(file_name, "rb") as f:
            image = Image.open(f)
            # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            image = utils._apply_exif_orientation(image)
            # image.save("temp_OG.png")    
            image = image.convert(self.img_format)
            image = self.perturb(image)
            # image.save(f"temp-{self.mode}.png")

            return utils.convert_PIL_to_numpy(image, self.img_format)

    
    def other_setups(self, dataset_dict, image):
        utils.check_image_size(dataset_dict, image)
        
        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w
        W_wop = image_shape[1] - np.sum(padding_mask[0, :])
        H_wop = image_shape[0] - np.sum(padding_mask[:, 0])
        image_shape_wop = (H_wop, W_wop) # without padding
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))


        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            # instances = utils.annotations_to_instances(annos, image_shape)
            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            
            if 'gt_masks' in instances._fields.keys():
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            # instances = utils.filter_empty_instances(instances)
            instances,_mask = utils.filter_empty_instances(instances, return_mask=True)
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            # if hasattr(instances, 'gt_masks'):
            #     gt_masks = instances.gt_masks
            #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            #     instances.gt_masks = gt_masks
            # NOTE: Here we get the size of image without padding. 
            # This is different from the original Mask2Former
            setattr(instances, "_image_size", image_shape_wop)
            dataset_dict["instances"] = instances

        return dataset_dict


    def image_call(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = self.load_image(dataset_dict)

        dataset_dict = self.other_setups(dataset_dict, image)

        
        image = dataset_dict['image']
        # print(image.shape)
        image = self.post_perturb(image)
        # save_image( normalize( image ), f"temp-{self.mode}.png" )
        dataset_dict['image'] = image

        # save_image( normalize( image ), f"temp-{self.mode}.png" )
        # quit()

        return dataset_dict

        
        
        