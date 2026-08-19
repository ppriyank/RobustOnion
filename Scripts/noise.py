import torch 

import time
import os 
import sys 
import json 
from PIL import Image
import albumentations as A
import numpy as np 
# import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.utils import save_image

Root = "/data/priyank/synthetic/flickr_dataset_30k/"
intermediate = "flickr30k/flickr30k-images/"

subset = "DATASET/LR_flickr_separateGT_train_subset3.json"


def normalize(x):return (x - x.min()) / (x.max() - x.min())



# Read the JSON file
with open(subset, 'r') as file:
    data = json.load(file)



focus_blur = A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=1)
chromatic_abb = A.ChromaticAberration(primary_distortion_limit=(-0.2, 0.2), secondary_distortion_limit=(-0.5, 0.5), mode='green_purple', interpolation=1, p=1)
iso_abb = A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.4, 0.7), p=1)
jpeg_abb = A.ImageCompression (compression_type='jpeg', quality_range=(40, 50), p=1)
pixel_abb = A.PixelDropout (dropout_prob=0.1, per_channel=False, drop_value=0, mask_drop_value=None, p=1)
fog_ab = A.RandomFog(alpha_coef=0.1, fog_coef_range=(0.3, 1), p=1)
salt_pepper_ab = A.SaltAndPepper (amount=(0.01, 0.06), salt_vs_pepper=(0.4, 0.6), p=1) 

transform_fn = transforms.ToTensor()


focus_fn = False  
chromatic_abb_fn = False 
iso_noise_fn = False  
jpeg_abb_fn = False 
pixel_abb_fn = False  
fog_ab_fn = False  
salt_pepper_ab_fn = False  



motion_fn = True     
rain_fn = False 
snow_ab_fn = False 
atmospheric_turbulence = False

low_res_fourier = False
low_res = False     
 


####### low resolution 
if low_res:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        # os.path.exists(path)
        image = Image.open(path)
        
        image = transform_fn(image)
        H, W = image.shape[1:]
        resize_transform1 = transforms.Resize( (H // 8, W // 8 ) , antialias=False) 
        resize_transform2 = transforms.Resize((H , W) , antialias=False) 
        image = resize_transform2(resize_transform1(image))

        save_image(normalize(image), "temp_LR.png")
        quit()
    # Elapsed time: 89.55751705169678 seconds

####### Salt pepper noise
if salt_pepper_ab_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = salt_pepper_ab(image =np.array(image))["image"]
        image = Image.fromarray(image)
    # Elapsed time: 39.882861852645874 seconds        

####### ISO nosie   
if iso_noise_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = iso_abb(image =np.array(image))["image"]
        image = Image.fromarray(image)
    # Elapsed time: 108.9937093257904 seconds

####### focus blur   
if focus_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = focus_blur(image =np.array(image))["image"]
        image = Image.fromarray(image)
        image.save("temp_focus.png")
        quit()
    # Elapsed time: 86.60300135612488 seconds

####### chrmoatic abbrevation   
if chromatic_abb_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = chromatic_abb(image =np.array(image))["image"]
        image = Image.fromarray(image)
    # Elapsed time: 40.865559339523315 seconds
        
####### JPG compression
if jpeg_abb_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = jpeg_abb(image =np.array(image))["image"]
        image = Image.fromarray(image)
    # Elapsed time: 30.216266632080078 seconds

####### Fog
if fog_ab_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = fog_ab(image =np.array(image))["image"]
        image = Image.fromarray(image)
        image.save("temp.png")
        quit()
    # Elapsed time: 925.8988969326019 seconds

####### Pixel drop
if pixel_abb_fn:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        image = pixel_abb(image =np.array(image))["image"]
        image = Image.fromarray(image)
    # Elapsed time: 49.91349506378174 seconds


####### motion_blur  
if motion_fn:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current_dir)
    sys.path.append(os.path.join(root))
    sys.path.append(os.path.join(root, 'Weather_Simulation'))
    
    from Weather_Simulation.motionblur_effect import Motion_Blur_Generator, Pickable_Motion_Blur_Generator
    perturb = Motion_Blur_Generator()


    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        # os.path.exists(path)
        
        image = Image.open(path)
        image = perturb(image)
        image.save("temp_MB.png")
        quit()
    # Elapsed time: 25.498008489608765 seconds

####### rain 
if rain_fn:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current_dir)
    sys.path.append(os.path.join(root))
    sys.path.append(os.path.join(root, 'Weather_Simulation'))
    
    from Weather_Simulation.weather import RainEffectGenerator2
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    perturb = RainEffectGenerator2(device=device)

    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        # os.path.exists(path)
        image = Image.open(path)
        image = perturb(image)
        image.save("temp.png")
        quit()
    
    
####### SNOW
if snow_ab_fn:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current_dir)
    sys.path.append(os.path.join(root))
    sys.path.append(os.path.join(root, 'Weather_Simulation'))
    
    from Weather_Simulation.weather import SnowEffectGenerator2        
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    perturb = SnowEffectGenerator2(device=device)

    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        
        image = perturb(image)
        image.save("temp.png")
        quit()
        
        







####### atmospheric turbulence 

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


if atmospheric_turbulence:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    import torch 
    from torchvision import transforms
    from TurbulenceSimulatorPython.turbStats import tilt_mat, corr_mat, get_r0
    from TurbulenceSimulatorPython.simulator import Simulator
    from TurbulenceSimulatorPython.helper import factorixze , normalize


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # Set turbulence parameters
    size = 512
    N = 512  # Image size
    D = 0.1  # Aperture diameter
    r0 = 0.05  # Fried parameter
    L = 3000  # Propagation distance
    # Run tilt_mat function
    tilt_mat(N, D, r0, L, save_path=f'{parent_dir}/TurbulenceSimulatorPython/data')
    # Define image transformations
    resize_transform1 = transforms.Resize( (size, size ) )
    
    correlation = -0.1 # [-0.1, -0.01, -1, -5]
    simulator = Simulator(D/r0, img_size=512, corr=correlation, data_path=f'{parent_dir}/TurbulenceSimulatorPython/data', device=device).to(device, dtype=torch.float32)
    simulator.eval()

    post_perturb = apply_atmospheric_perturb(simulator=simulator, resize_transform1=resize_transform1, device=device)
    
    
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        
        image = transform_fn(image)
        image = post_perturb(image)
        save_image(normalize(image), "temp.png")

        quit()







def fft(channel):
    fft = np.fft.fft2(channel)
    f_transform_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(f_transform_shifted) + 1
    magnitude_spectrum = np.log1p(magnitude_spectrum)
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    magnitude_spectrum *= 255.0 
    return magnitude_spectrum

if low_res_fourier:
    start_time = time.time()
    for file in data['images']:
        path = os.path.join(Root, intermediate, file['file_name'])
        image = Image.open(path)
        # image= image.convert("L")
        H,W  = image.size

        channels = image.split() 
        result_array = np.zeros_like(image)
        for i, channel in enumerate(channels):
            result_array[..., i] = fft(channel)
        result_image = Image.fromarray(result_array)
        result_image.save("temp-OG.png")

        image = image.resize((H // 4, W // 4), Image.BICUBIC)
        image = image.resize((H, W ), Image.BICUBIC)

        channels = image.split() 
        result_array = np.zeros_like(image)
        for i, channel in enumerate(channels):
            result_array[..., i] = fft(channel)
        result_image = Image.fromarray(result_array)
        result_image.save("temp-FT.png")


        
        
        
        
        
        
        
        quit()


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")





        
        
        
        
    

# https://albumentations.ai/docs/api_reference/augmentations/transforms/
# cd ~/robustness_object_detection/
# conda activate OD
# python Scripts/noise.py


