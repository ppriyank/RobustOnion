import torch 
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
import random 

from torchvision.transforms.functional import to_pil_image
import os, sys
import albumentations as A
from mmengine.dist import get_dist_info

def low_resolution_preprocess(img, sev=2 ** 2):
    H, W = img.size
    img = img.resize(( int(H // sev) , int(W // sev) ), resample=0).resize( (H,W), resample=2)
    return img 


def low_resolution2(img):
    H, W = img.shape[1:]
    resize_transform1 = transforms.Resize( (H // 8, W // 8 ) , antialias=False) 
    resize_transform2 = transforms.Resize((H , W) , antialias=False) 
    img = resize_transform2(resize_transform1(img))
    return img 

def low_resolution_MULTI(img):
    H, W = img.shape[1:]
    ratio = random.uniform(1/12, 1/5)
    resize_transform1 = transforms.Resize( ( int(H * ratio) , int(W * ratio)  ) , antialias=False) 
    resize_transform2 = transforms.Resize((H , W) , antialias=False) 
    img = resize_transform2(resize_transform1(img))
    return img 


def normalize(x):return (x - x.min()) / (x.max() - x.min())


class apply_perturb:
    def __init__(self, perturb_fn=None, seed=None):
        self.perturb_fn = perturb_fn
        if seed is not None:
            self.perturb_fn = A.Compose([perturb_fn], seed=seed)

    def __call__(self, image):
        image = self.perturb_fn(image = image )["image"]
        return image
    


class apply_pil_perturb_perturb:
    def __init__(self, perturb_fn=None):
        self.perturb_fn = perturb_fn
    
    def __call__(self, image):
        image = Image.fromarray(image).convert(mode="RGB")
        return self.perturb_fn(image)

        




class apply_atmospheric_perturb:
    def __init__(self, resize_transform1=None, simulator=None, device=None):
        self.resize_transform1 = resize_transform1
        self.simulator = simulator 
        self.device = device 

    def __call__(self, image):
        # image.max(), image.min()
        # save_image(normalize(image), "temp_og.png"); save_image(image, "temp_og.png")
        H,W = image.shape[1:]
        image = self.resize_transform1(image.float())
        # image = image.cuda().cpu()
        image = image.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            image = self.simulator(image.unsqueeze(0)).detach().cpu()
        image = transforms.Resize( (H, W) )(image)
        # save_image(normalize(image), "temp.png"); save_image(image, "temp.png")
        image = image.to(torch.uint8)
        return image

    
        
        
        
        
        


def do_nothing(x):
    return x



class apply_multiple_noises:
    def __init__(self, ONE_NOISE=None):
        self.noises = {}
        self.ONE_NOISE = ONE_NOISE

    def add_noise(self, noise, perturb_local, post_perturb_local):
        self.noises[noise] = {
            'perturb_local' : perturb_local, 
            'post_perturb_local': post_perturb_local
        }

    def setup(self, ):
        self.keys = list(self.noises.keys())

    def pre_method(self, x):
        if self.ONE_NOISE:
            noise = random.choice(self.keys)
            x = self.noises[noise]['perturb_local'](x)
            return x, noise 
        else:
            random.shuffle(self.keys)
            for noise in self.keys:
                x = self.noises[noise]['perturb_local'](x)
            return x, None 
            
    def post_method(self, x, noise):
        if self.ONE_NOISE:
            x = self.noises[noise]['post_perturb_local'](x)
            return x
        else:
            random.shuffle(self.keys)
            for noise in self.keys:
                x = self.noises[noise]['post_perturb_local'](x)
            return x
            



def display_boxes(image, box, border_width = 5, border_color = 'red', prefix_name="temp", categories=None, font=20, mode="single"):
    save_image(image, f"{prefix_name}_OG.png")
    drawn_images = []
    
    image = to_pil_image(image)
    # image.save(f"{prefix_name}_{i}_raw.png")
    draw = ImageDraw.Draw(image)
    
    # type(box)
    if hasattr(box, "tensor"):
        box = box.tensor
    # assert anno[i].mode == "xyxy"
    for count, rectangle_coords in enumerate(box):
        ## xmin , ymin , xmax, ymax
        for k in range(border_width):
            draw.rectangle( [rectangle_coords[0].item() - k, rectangle_coords[1].item() - k, rectangle_coords[2].item() + k, rectangle_coords[3].item() + k], outline=border_color )

    image.convert('RGB').save(f"{prefix_name}_boxes.png")




def display_box_over_img(img, anno, labels=None, mapper=None, border_width = 5, mode="xywh", direct_box=False):
    color_indicator = {0: 'red', 1:'blue', 2:'yellow', 3:'green', 4:'orange', 5:'purple', 6:'brown', 7:'pink', 8:'black'}
    img = to_pil_image(img)
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
            text = mapper[labels[k]]
            
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
            if text:
                position = rectangle_coords[0] - border_width , rectangle_coords[1] - border_width
                draw.text(position, text, fill='white')
                
    img.save("temp2.png")








def pertubation_methods(mode, train_mode=False , seed=None):
    post_perturb = do_nothing
    perturb = do_nothing
    print(f" \n\n USING ... {mode} \n\n")
    if mode == "pixel_dropout":
        perturb_fn = A.PixelDropout (dropout_prob=0.1, per_channel=False, drop_value=0, mask_drop_value=None, p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    elif mode == "salt_pepper":
        perturb_fn = A.SaltAndPepper (amount=(0.06, 0.1), salt_vs_pepper=(0.4, 0.6), p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    elif mode == "iso_blur":
        perturb_fn = A.ISONoise(color_shift=(0.08, 0.5), intensity=(0.4, 1), p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    

    elif mode == "low-res" or mode == "low-res_MULTI":
        if mode == "low-res_MULTI":
            post_perturb = low_resolution_MULTI
        else:
            post_perturb = low_resolution2
    
    elif mode == "focus_blur":
        perturb_fn = A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    elif mode == "chromatic":
        perturb_fn = A.ChromaticAberration(primary_distortion_limit=(-1, 1), secondary_distortion_limit=(-2, 2), mode='green_purple', interpolation=1, p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    elif mode == "jpg_compression":
        if train_mode:
            perturb_fn = A.ImageCompression (compression_type='jpeg', quality_range=(2, 30), p=1)
        else:
            perturb_fn = A.ImageCompression (compression_type='jpeg', quality_range=(10, 20), p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    elif mode == "motion_blur2" or mode == "motion_blur2_MULTI" or mode == "Pickable_Motion_Blur_Generator":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))

        from Weather_Simulation.motionblur_effect import Motion_Blur_Generator, Pickable_Motion_Blur_Generator
        
        perturb_fn = Pickable_Motion_Blur_Generator(back_to_pil=False )
        # if mode == "Pickable_Motion_Blur_Generator":
            
        # else:
        #     perturb_fn = Motion_Blur_Generator(back_to_pil=False )
        perturb = apply_pil_perturb_perturb(perturb_fn)
        
            
        
        
    
    elif mode == "fog":
        perturb_fn = A.RandomFog(alpha_coef=0.1, fog_coef_range=(0.3, 1), p=1)
        perturb = apply_perturb(perturb_fn, seed=seed)
    elif mode == "rain_model2":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))
        from Weather_Simulation.weather import RainEffectGenerator2
            
        rank, world_size = get_dist_info()
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        perturb = RainEffectGenerator2(back_to_pil=False, device=device)
        perturb = apply_pil_perturb_perturb(perturb)

    elif mode == "snow_model2":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.append(parent_dir)
        sys.path.append(os.path.join(parent_dir, "Weather_Simulation"))
        from Weather_Simulation.weather import SnowEffectGenerator2
            
        rank, world_size = get_dist_info()
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        perturb = SnowEffectGenerator2(back_to_pil=False, device=device)
        perturb = apply_pil_perturb_perturb(perturb)


    elif mode == "atmospheric":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # '~/robustness_object_detection/MMGDINO/proposed_modules'
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        # '~/robustness_object_detection'
        sys.path.append(parent_dir)
        
        from TurbulenceSimulatorPython.turbStats import tilt_mat, corr_mat, get_r0
        from TurbulenceSimulatorPython.simulator import Simulator
        from TurbulenceSimulatorPython.helper import factorixze , normalize

        rank, world_size = get_dist_info()
        # Set turbulence parameters
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        size = 512
        N = 512  # Image size
        D = 0.1  # Aperture diameter
        r0 = 0.05  # Fried parameter
        L = 3000  # Propagation distance
        # Run tilt_mat function
        tilt_mat(N, D, r0, L, save_path=f'{parent_dir}/TurbulenceSimulatorPython/data')
        resize_transform1 = transforms.Resize( (size, size ) )

        correlation = -0.1 # [-0.1, -0.01, -1, -5]
        # simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{parent_dir}/TurbulenceSimulatorPython/data', device=device).to(device, dtype=torch.float32)
        simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{parent_dir}/TurbulenceSimulatorPython/data', device=device).to(device, dtype=torch.float32)
        simulator.eval()

        post_perturb = apply_atmospheric_perturb(simulator=simulator, resize_transform1=resize_transform1, device=device)
    
    elif mode == 'NONE' :
        _ = 0 
    elif "*" in mode:
        return do_nothing, do_nothing 
    elif mode in ["rain_model3", "snow_model3", "atmospheric2"]:
        _ = 0
    else:
    
        import pdb
        pdb.set_trace()
    return perturb, post_perturb           



    
    