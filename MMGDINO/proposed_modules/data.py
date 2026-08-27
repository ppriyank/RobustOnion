import os 
import sys 
import torch
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.lvis import LVISV1Dataset
from mmdet.registry import DATASETS
import mmcv 
from torchvision.utils import save_image
from torchvision.transforms import transforms

from PIL import Image

from mmengine.dataset.base_dataset import Compose
import albumentations as A

NONE_IMPORT = None 

FILE_LAODER = mmcv.transforms.loading.LoadImageFromFile


from .perturb_util import low_resolution2, apply_perturb, apply_pil_perturb_perturb, apply_atmospheric_perturb, \
    do_nothing, save_image, normalize, pertubation_methods, display_box_over_img
# apply_multiple_noises, low_resolution_MULTI, low_resolution_preprocess

from mmengine.dist import get_dist_info


# https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/base_det_dataset.py#L13
# https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/coco.py
@DATASETS.register_module()
class COCODataset_Perturb(CocoDataset):

    def __init__(self, mode=None, severity=None, **kwargs):
        super().__init__(**kwargs) 
        assert self.test_mode is True 
        assert type(self.pipeline.transforms[0]) == FILE_LAODER
        
        self.img_loader = self.pipeline.transforms[0]
        self.remaining_pipeline = Compose(transforms=None)
        self.remaining_pipeline.transforms = self.pipeline.transforms[1:]
        
        self.perturb_setup(mode, severity)
        self.__getitem__(0)

    def perturb_setup(self, mode, severity=None):
        self.mode = mode 
        self.post_perturb = do_nothing
        self.perturb = do_nothing
        print(f" \n\n USING ... {mode} \n\n")
        
        if mode == "low-res":
            self.perturb = lambda x : x 
            self.post_perturb = self.low_resolution2
            self.severity = None
            if severity is not None:
                self.severity = 2 ** severity 

            
        elif mode == "focus_blur":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=1)], seed=0)
        elif mode == "gaussian_blur":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.GaussianBlur(blur_limit=0, sigma_limit=(2.0, 5.0), p=1)], seed=0)
        elif mode == "chromatic":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.ChromaticAberration(primary_distortion_limit=(-1, 1), secondary_distortion_limit=(-2, 2), mode='green_purple', interpolation=1, p=1)], seed=0)
        elif mode == "iso_blur":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.ISONoise(color_shift=(0.08, 0.5), intensity=(0.4, 1), p=1)], seed=0)
        elif mode == "jpg_compression":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.ImageCompression (compression_type='jpeg', quality_range=(10, 20), p=1)], seed=0)
        elif mode == "pixel_dropout":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.PixelDropout (dropout_prob=0.1, per_channel=False, drop_value=0, mask_drop_value=None, p=1)], seed=0)
        elif mode == "fog":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.RandomFog(alpha_coef=0.1, fog_coef_range=(0.3, 1), p=1)], seed=0)
        elif mode == "salt_pepper":
            self.perturb = self.apply_perturb
            self.perturb_fn = A.Compose([A.SaltAndPepper (amount=(0.06, 0.1), salt_vs_pepper=(0.4, 0.6), p=1)], seed=0)
        elif mode == "snow_model2" or mode == "snow_model3":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            sys.path.append(parent_dir)
            sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))
            from Weather_Simulation.weather import SnowEffectGenerator2
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

            self.perturb_fn = SnowEffectGenerator2(back_to_pil=False, device=device)
            self.perturb = self.apply_pil_perturb_perturb

        elif mode == "rain_model2" or mode == "rain_model3":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            sys.path.append(parent_dir)
            sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))
            from Weather_Simulation.weather import RainEffectGenerator2
            
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

            self.perturb_fn = RainEffectGenerator2(back_to_pil=False, device=device)
            self.perturb = self.apply_pil_perturb_perturb
            
        elif mode == "atmospheric" or mode == "atmospheric2":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # '~/robustness_object_detection/MMGDINO/proposed_modules'
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            # '~/robustness_object_detection'
            sys.path.append(parent_dir)
            
            from TurbulenceSimulatorPython.turbStats import tilt_mat, corr_mat, get_r0
            from TurbulenceSimulatorPython.simulator import Simulator
            from TurbulenceSimulatorPython.helper import factorixze , normalize

            # Set turbulence parameters
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            size = 512
            N = 512  # Image size
            D = 0.1  # Aperture diameter
            r0 = 0.05  # Fried parameter
            L = 3000  # Propagation distance
            # Run tilt_mat function
            tilt_mat(N, D, r0, L, save_path=f'{parent_dir}/TurbulenceSimulatorPython/data')
            self.resize_transform1 = transforms.Resize( (size, size ) )

            correlation = -0.1 # [-0.1, -0.01, -1, -5]
            self.simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{parent_dir}/TurbulenceSimulatorPython/data', device=self.device).to(self.device, dtype=torch.float32)
            self.simulator.eval()

            # self.perturb = lambda x : x 
            self.perturb = do_nothing
            
            self.post_perturb = self.apply_atmospheric_perturb
        
        elif mode == "motion_blur2":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            sys.path.append(parent_dir)
            sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))

            # from Weather_Simulation.motionblur_effect import Motion_Blur_Generator
            # self.perturb_fn = Motion_Blur_Generator(back_to_pil=False )
            from Weather_Simulation.motionblur_effect import Motion_Blur_Generator, Pickable_Motion_Blur_Generator
            self.perturb_fn = Pickable_Motion_Blur_Generator(back_to_pil=False )
            self.perturb = self.apply_pil_perturb_perturb
        else:
            '''
            do nothing 
            '''
            
    def low_resolution2(self, img):
        H, W = img.shape[1:]
        if self.severity is not None:
            resize_transform1 = transforms.Resize( (H // self.severity, W // self.severity ) , antialias=False) 
        else:
            resize_transform1 = transforms.Resize( (H // 8, W // 8 ) , antialias=False) 
        resize_transform2 = transforms.Resize((H , W) , antialias=False) 
        img = resize_transform2(resize_transform1(img))
        # save_image(img, "temp.png")
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        return img  

    def apply_perturb(self, image):
        image = self.perturb_fn(image = image )["image"]
        return image
    
    def apply_pil_perturb_perturb(self, image):
        image = Image.fromarray(image).convert(mode="RGB")
        return self.perturb_fn(image)
    
    def apply_atmospheric_perturb(self, image):
        # image.max(), image.min()
        # save_image(normalize(image.float()), "temp_og.png")
        H,W = image.shape[1:]
        image = self.resize_transform1(image.float())
        # image = image.cuda().cpu()
        image = image.to(self.device, dtype=torch.float32)
        image = self.simulator(image.unsqueeze(0)).detach().cpu()
        image = transforms.Resize( (H, W) )(image)
        # save_image(normalize(image), "temp.png")
        image = image.to(torch.uint8)
        return image

    def prepare_data(self, idx):
        
        data_info = self.get_data_info(idx)
        info = self.img_loader(data_info)
        img = info['img']
        img = self.perturb(img)
        info['img'] = img 
        
        return self.remaining_pipeline(info)

    def __getitem__(self, idx):
        data = self.prepare_data(idx)
        
        img = data['inputs']
        img = self.post_perturb(img)
        data['inputs'] = img 
        
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # quit()

        return data
        
@DATASETS.register_module()
class LVISDataset_Perturb(LVISV1Dataset):

    def __init__(self, mode=None, **kwargs):
        super().__init__(**kwargs) 
        assert self.test_mode is True 
        assert type(self.pipeline.transforms[0]) == FILE_LAODER
        
        self.img_loader = self.pipeline.transforms[0]
        self.remaining_pipeline = Compose(transforms=None)
        self.remaining_pipeline.transforms = self.pipeline.transforms[1:]
        
        self.mode = mode 
        self.perturb, self.post_perturb = pertubation_methods(mode, seed=0)
        self.__getitem__(0)


    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        info = self.img_loader(data_info)
        img = info['img']
        img = self.perturb(img)
        info['img'] = img 
        return self.remaining_pipeline(info)

    def __getitem__(self, idx):
        data = self.prepare_data(idx)
        img = data['inputs']
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"PRE-{self.mode}.png" )

        img = self.post_perturb(img)
        data['inputs'] = img 
        
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # quit()

        return data



# from mmengine.dataset import BaseDataset
# mmdet.datasets.coco.CocoDataset
@DATASETS.register_module()
class Extern_Dataset(CocoDataset):

    def __init__(self, fog_level=None, **kwargs):
        self.process_path = False 
        if 'bdd100k' in kwargs['ann_file']:
            kwargs['metainfo'] = {'classes': ("person", "car" , "rider", "bus", "truck", "bike", "motor", "traffic light", "traffic sign")}
        elif 'DAWN' in kwargs['ann_file']:
            kwargs['metainfo'] = {'classes': ("car", "person", "bicycle", "motorcycle", "truck",  "bus")}
        elif 'WEDGE' in kwargs['ann_file']:
            kwargs['metainfo'] = {'classes': ("car", "person", "motorcycle", "bicycle", "van" ,"truck", "bus")}          
        elif "Cityscape" in kwargs['ann_file']:
            kwargs['metainfo'] = {'classes': ("car", "bicycle", "bus", "motorcycle", "truck")}
            self.level=fog_level
            self.process_path = True 
        
        super().__init__(**kwargs) 
        namer = {y:self.metainfo['classes'][y] for x,y in self.cat2label.items()}                
        idx=0
        # self.debug(idx, namer)
                
    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if self.process_path:
            filename = data_info['img_path']
            filename = filename + f'_leftImg8bit_foggy_beta_{self.level}.png'
            data_info['img_path'] = filename            
        return self.pipeline(data_info)

    def debug(self, idx=0, namer=None):
        data = self.__getitem__(idx)
        print("====", data['data_samples'].img_path)
        img = data['inputs']
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"ext-og.png" )
        boxes = data['data_samples'].gt_instances.bboxes
        
        orig_h, orig_w = data['data_samples'].ori_shape
        new_h, new_w = data['data_samples'].img_shape

        boxes.rescale_(( new_h / orig_h , new_w/ orig_w))

        labels = data['data_samples'].gt_instances.labels
        display_box_over_img(img, anno=boxes.tensor, labels=labels.tolist(), mapper=namer, border_width = 5, mode="xyxy", direct_box=True)
        
        














    
    
   
   
# https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L120

# https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/datasets/odvg.py#L13
from mmdet.datasets.odvg import ODVGDataset

@DATASETS.register_module()
class ODVGDataset_Peturb(ODVGDataset):
    def __init__(self, mode=None, **kwargs) -> None:
        super().__init__(**kwargs)
        
        assert self.test_mode == False 
        
        self.mode = mode     
        self.perturb, self.post_perturb = pertubation_methods(mode)
        self.img_loader = self.pipeline.transforms[0]

        self.remaining_pipeline = Compose(transforms=None)
        self.remaining_pipeline.transforms = self.pipeline.transforms[1:]
        print(f"\nSize of Training Data ... {kwargs['ann_file']} {self.__len__()}\n")
        idx = 0 
        self.__getitem__(idx)
        
    def load_data_list(self):
        ### Not official flicker but just how we structured out dataset (apologies)
        self.ann_file = ".." + self.ann_file.split("..")[-1]
        self.data_prefix['img'] = self.data_prefix['img'].replace("flickr30k_images", "flickr30k-images")
        if "Perturbed" in self.data_prefix['img']:
            self.data_prefix['img'] = self.data_prefix['img'].replace("flickr30k_images", "").replace("flickr30k-images", "")
        return super().load_data_list()

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        
        info = self.img_loader(data_info)
        img = info['img']
        img = self.perturb(img)
        info['img'] = img 

        return self.remaining_pipeline(info)
    
    def __getitem__(self, idx: int) -> dict:
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue

        
        img = data['inputs']
        img = self.post_perturb(img)
        data['inputs'] = img 

        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-{self.mode}.png" )
        # quit()

        return data 




from mmengine.fileio import get_local_path
import json
import os.path as osp


@DATASETS.register_module()
class Extern_Dataset_CoCo_train(CocoDataset):

    def __init__(self, **kwargs):
        kwargs['metainfo'] = {'classes': ("car", "person", "motorcycle", "bicycle", "van" ,"truck", "bus")}          
        super().__init__(**kwargs) 
        namer = {y:self.metainfo['classes'][y] for x,y in self.cat2label.items()}                
        idx=0
        x = self.__getitem__(idx)
        
        self.debug(idx, namer)
    
                
    def debug(self, idx=0, namer=None):
        data = self.__getitem__(idx)
        img = data['inputs']
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"ext-og.png" )
        boxes = data['data_samples'].gt_instances.bboxes
        
        labels = data['data_samples'].gt_instances.labels
        display_box_over_img(img, anno=boxes.tensor, labels=labels.tolist(), mapper=namer, border_width = 5, mode="xyxy", direct_box=True)
        
