import os 

import numpy as np 
import torch 
from skimage import color
import random
from PIL import Image


from lib.rain_gen import RainGenUsingNoise
from lib.snow_gen import SnowGenUsingNoise

from lib.gan_networks import define_G
import torchvision.transforms as transforms
from torchvision.utils import save_image


def normalize(x):return (x - x.min()) / (x.max() - x.min())


class scale_width_transform():
    def __init__(self, load_size, crop_size):
        self.target_size = load_size
        self.crop_size = crop_size
        self.method = Image.BICUBIC

    def __call__(self, img):
        ow, oh = img.size
        if ow == self.target_size and oh >= self.crop_size:
            return img
        w = self.target_size
        h = int(max(self.target_size * oh / ow, self.crop_size))
        return img.resize((w, h), self.method)


class Weather_Generator:

    def __init__(self, back_to_pil=True, device=None, wt = "Gan_Wts/clear2rainy.pth" ):
        # Creating model
        input_nc = 3
        output_nc = 3
        ngf = 64
        netG = 'resnet_9blocks'
        norm = 'instance'
        no_dropout = True
        init_type = 'normal'
        init_gain = 0.02
        gpu_ids = []
    
        current_dir = os.path.dirname(os.path.abspath(__file__))

        self.netG_A = define_G(input_nc, output_nc, ngf, netG, norm, not no_dropout, init_type, init_gain, gpu_ids)
        chkpntA = torch.load( os.path.join(current_dir, wt) ) 
        self.netG_A.load_state_dict(chkpntA)
        self.netG_A.eval()
        if device :
            self.netG_A = self.netG_A.to(device,dtype=torch.float32)
        else:
            self.netG_A = self.netG_A.cuda()
        self.device = device
        load_size = 1280
        crop_size = 224
        self.transform = self.get_transform(load_size=load_size, crop_size=crop_size)
        self.back_to_pil = back_to_pil

    def __scale_width(self, img, target_size, crop_size):
        method = Image.BICUBIC
        ow, oh = img.size
        if ow == target_size and oh >= crop_size:
            return img
        w = target_size
        h = int(max(target_size * oh / ow, crop_size))
        return img.resize((w, h), method)

    def get_transform(self, load_size, crop_size):
        # transform_list = [transforms.Lambda(lambda img: self.__scale_width(img, load_size, crop_size)),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_list = [scale_width_transform(load_size=load_size, crop_size=crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def __call__(self, img):
        ow, oh = img.size
        img = self.transform(img)
        # save_image( normalize( img ), f"temp-OG.png" )
        # save_image( normalize( img[torch.tensor((2, 1, 0))] ), f"temp-OG.png" )

        img = img.unsqueeze(0)#.to('cuda')
        img = img.to(self.device,dtype=torch.float32)
        # print("=======", img.device, self.netG_A.model[1].weight.device)
        with torch.no_grad():
            out = self.netG_A(img)
        out = out.squeeze()

        resize_transform1 = transforms.Resize( (oh, ow) )
        out= resize_transform1(out)
        # save_image( normalize(out), "temp.png" )
        # save_image( out, "temp.png" )

        if self.back_to_pil:
            out = Image.fromarray( (normalize(out) * 255).cpu().permute(1, 2, 0).numpy().astype('uint8') )
            # out.save("temp.png")
        else:
            out = (normalize(out) * 255).cpu().permute(1, 2, 0).numpy().astype('uint8')
        
        
        return out 
    
    



class RainEffectGenerator2 (Weather_Generator):
    def __init__(self, back_to_pil=True, device=None, wt = "Gan_Wts/clear2rainy.pth" ):
        super().__init__(back_to_pil=back_to_pil, device=device, wt=wt)

        
    
    
class SnowEffectGenerator2(Weather_Generator):
    def __init__(self, back_to_pil=True, device=None, wt = "Gan_Wts/clear2snowy.pth" ):
        super().__init__(back_to_pil=back_to_pil, device=device, wt=wt)

    