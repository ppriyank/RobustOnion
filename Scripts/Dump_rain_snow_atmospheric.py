import os 
import json
from pycocotools.coco import COCO
import random 
import sys
from PIL import Image
import torch 
from torchvision.transforms import transforms


def _laod_images(file):
    if not os.path.exists(file):
        return set(), set()
    coco_api = COCO(file)
    IDS = list(coco_api.imgs.keys())
    images = set()
    images_ids = set()
    for e in IDS:
        img = coco_api.loadImgs(e)[0]
        images.add(img['file_name'])
        images_ids.add(img['id'])
    return images, images_ids


device = torch.device('cuda:1') 
DEST_ROOT = "/data/priyank/synthetic/flickr_dataset_30k/Perturbed"

RAIN_MODEL = False  
SNOW_MODEL = False  
ATMOSPHERE_MODEL = True  
# MOTION_DUMP = False # can be generated on the fly, dont dump 

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(current_dir)
sys.path.append(root)
sys.path.append(os.path.join(root, "Weather_Simulation"))

if RAIN_MODEL:
    from Weather_Simulation.weather import RainEffectGenerator2        
    
    perturb = RainEffectGenerator2(device=device)
    DEST = os.path.join(DEST_ROOT, "RAIN")        

if SNOW_MODEL:
    from Weather_Simulation.weather import SnowEffectGenerator2        
    perturb = SnowEffectGenerator2(device=device)
    DEST = os.path.join(DEST_ROOT, "SNOW")        



class apply_atmospheric_perturb:
    def __init__(self, resize_transform1=None, simulator=None, device=None):
        self.resize_transform1 = resize_transform1
        self.simulator = simulator 
        self.device = device 
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image):
        image = self.to_tensor(image)
        # image.max(), image.min()
        H,W = image.shape[1:]
        image = self.resize_transform1(image)
        image = image.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            image = self.simulator(image.unsqueeze(0)).detach().cpu()
        image = transforms.Resize( (H, W) )(image)
        image = self.to_pil(image)
        return image

if ATMOSPHERE_MODEL:
    from TurbulenceSimulatorPython.turbStats import tilt_mat, corr_mat, get_r0
    from TurbulenceSimulatorPython.simulator import Simulator
    from TurbulenceSimulatorPython.helper import factorixze , normalize
    

    size = 512
    N = 512  # Image size
    D = 0.1  # Aperture diameter
    r0 = 0.05  # Fried parameter
    L = 3000  # Propagation distance
    
    # Run tilt_mat function
    tilt_mat(N, D, r0, L, save_path=f'{root}/TurbulenceSimulatorPython/data')
    resize_transform1 = transforms.Resize( (size, size ) )

    correlation = -0.1 # [-0.1, -0.01, -1, -5]
    simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{root}/TurbulenceSimulatorPython/data', device=device).to(device, dtype=torch.float32)
    simulator.eval()
    perturb = apply_atmospheric_perturb(simulator=simulator, resize_transform1=resize_transform1, device=device)

    DEST = os.path.join(DEST_ROOT, "TURBULENCE")        

    

# MOTION_DUMP
# from Weather_Simulation.motionblur_effect import Motion_Blur_Generator, Pickable_Motion_Blur_Generator
# perturb = Motion_Blur_Generator()\
# DEST = os.spath.join(DEST_ROOT, "motion_blur")

   
        
        
        


all_imgs = "/data/priyank/synthetic/flickr_dataset_30k/flickr30k/flickr30k-images"
SUBSET="DATASET/final_flickr_separateGT_train_subset3.json"



no_of_files = len(os.listdir(all_imgs)) - 1 
train, train_ids = _laod_images(SUBSET)
No_of_images = len(train)

for i,img_name in enumerate(train):
    print(f"[{i} / {No_of_images}]", img_name, end="\r")
    img_path = os.path.join(all_imgs, img_name)    
    assert os.path.exists(img_path)
    image = Image.open(img_path)
    # image.save("temp_OG.png")
    original_size = image.size
    image = perturb(image)
    # image.save("temp_NOISE.png")
    # quit()
    perturbed__size = image.size
    assert  original_size == perturbed__size
    dest = os.path.join(DEST, img_name)
    image.save(dest)
    


# cd ~/robustness_object_detection/
# python Scripts/Dump_rain_snow_atmospheric.py