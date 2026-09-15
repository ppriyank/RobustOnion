import torch 
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np 

import os 
import sys
import albumentations as A





def normalize(x):return (x - x.min()) / (x.max() - x.min())

def do_nothing(x):
    return x


def low_resolution2(img, downsample_res = 8 ):
    H, W = img.shape[1:]
    resize_transform1 = transforms.Resize( (H // downsample_res, W // downsample_res ) , antialias=False) 
    resize_transform2 = transforms.Resize((H , W) , antialias=False) 
    img = resize_transform2(resize_transform1(img))
    return img 


class apply_low_res:
    def __init__(self, downsample_res = 8):
        self.downsample_res = downsample_res
        
    def __call__(self, img):
        H, W = img.shape[1:]
        resize_transform1 = transforms.Resize( (H // self.downsample_res, W // self.downsample_res ) , antialias=False) 
        resize_transform2 = transforms.Resize((H , W) , antialias=False) 
        img = resize_transform2(resize_transform1(img))
        return img 
        
    


class apply_perturb:
    def __init__(self, perturb_fn=None):
        self.perturb_fn = perturb_fn
    
    def __call__(self, image):
        image = self.perturb_fn(image = np.array(image) )["image"]
        image = Image.fromarray(image)
        return image
    

class apply_atmospheric_perturb:
    def __init__(self, resize_transform1=None, simulator=None, device=None):
        self.resize_transform1 = resize_transform1
        self.simulator = simulator 
        self.device = device 

    def __call__(self, image):
        # image.max(), image.min()
        # save_image(normalize(image), "temp_og.png"); save_image(image, "temp_og.png")
        H,W = image.shape[1:]
        image = self.resize_transform1(image)
        # image = image.cuda().cpu()
        image = image.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            image = self.simulator(image.unsqueeze(0)).detach().cpu()
        image = transforms.Resize( (H, W) )(image)
        # save_image(normalize(image), "temp.png"); save_image(image, "temp.png")
        return image




def do_nothing(x):
    return x



def pertubation_methods(mode, train_mode=False , sev=3, ):
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
    

    elif mode == "low-res":
        if sev == None:
            sev = 3 
        post_perturb = apply_low_res(2 ** sev)
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
    elif mode == "motion_blur2":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
        sys.path.append(root)
        sys.path.append(os.path.join(root, "Weather_Simulation"))

        from Weather_Simulation.motionblur_effect import Motion_Blur_Generator
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
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
        sys.path.append(root)

        sys.path.append(os.path.join(root, "Weather_Simulation"))
        from Weather_Simulation.weather import RainEffectGenerator2        
        
        device = torch.device(f'cuda:{get_rank()}') if torch.cuda.is_available() else torch.device('cpu')
        perturb = RainEffectGenerator2(device=device)
        
        
    
    elif mode == "snow_model2":

        from maskrcnn_benchmark.utils.dist import get_local_rank, get_rank
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
        sys.path.append(root)

        sys.path.append(os.path.join(root, "Weather_Simulation"))
        from Weather_Simulation.weather import SnowEffectGenerator2        
        device = torch.device(f'cuda:{get_rank()}') if torch.cuda.is_available() else torch.device('cpu')
        perturb = SnowEffectGenerator2(device=device)
    
    
    elif mode == "atmospheric":
        
        from maskrcnn_benchmark.utils.dist import get_local_rank, get_rank
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))))
        sys.path.append(root)

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
        tilt_mat(N, D, r0, L, save_path=f'{root}/TurbulenceSimulatorPython/data')
        resize_transform1 = transforms.Resize( (size, size ) )

        correlation = -0.1 # [-0.1, -0.01, -1, -5]
        simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{root}/TurbulenceSimulatorPython/data', device=device).to(device, dtype=torch.float32)
        simulator.eval()

        post_perturb = apply_atmospheric_perturb(simulator=simulator, resize_transform1=resize_transform1, device=device)
    
    elif mode == 'NONE':
        _ = 0 


    else:
        import pdb
        pdb.set_trace()
    return perturb, post_perturb           








def display_box_over_img(img, anno, labels=None, mapper=None, border_width = 5, mode="xywh", direct_box=False):
    color_indicator = {0: 'red', 1:'blue', 2:'yellow', 3:'green', 4:'orange', 5:'purple', 6:'brown', 7:'pink', 8:'black'}
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for k,e in enumerate(anno):
        print(e)
        ## xmin , ymin , width, height
        if direct_box :
            rectangle_coords = e
            border_color = 'red' 
            if labels:
                border_color = color_indicator[labels[k]]
                
        else:
            rectangle_coords = e['bbox']    
            border_color = color_indicator[e['category_id']]
        if mapper:
            text = mapper[e['category_id']]
            
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
            if text:
                position = rectangle_coords[0] - border_width , rectangle_coords[1] - border_width
                draw.text(position, text, fill='white')
                
    img.save("temp2.png")

