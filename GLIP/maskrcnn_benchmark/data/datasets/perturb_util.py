import torch 
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw
import numpy as np 
import random 
import os 

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



class apply_multiple_noises:
    def __init__(self, ONE_NOISE=None):
        self.noises = {}
        self.ONE_NOISE = ONE_NOISE
        self.LABEL_NOISE = None 

    def add_noise(self, noise, perturb_local, post_perturb_local):
        self.noises[noise] = {
            'perturb_local' : perturb_local, 
            'post_perturb_local': post_perturb_local
        }

    def setup(self, ):
        self.keys = list(self.noises.keys())

    def list_noises(self,):
        self.setup( )
        return self.keys

    def pre_method(self, x, label_noise=None):
        if self.ONE_NOISE:
            noise = random.choice(self.keys)
            x = self.noises[noise]['perturb_local'](x)
            return x, noise 
        elif self.LABEL_NOISE:
            x = self.noises[label_noise]['perturb_local'](x)
            return x, label_noise 
        else:
            random.shuffle(self.keys)
            for noise in self.keys:
                x = self.noises[noise]['perturb_local'](x)
            return x, None 
            
    def post_method(self, x, noise):
        if self.ONE_NOISE or self.LABEL_NOISE:
            x = self.noises[noise]['post_perturb_local'](x)
            return x
        else:
            random.shuffle(self.keys)
            for noise in self.keys:
                x = self.noises[noise]['post_perturb_local'](x)
            return x
            




def display_box_over_img(img, anno, labels =None, border_width = 5, mode="xywh", direct_box=False, name='temp'):
    text = None 
    color_indicator = {0: 'red', 1:'blue', 2:'yellow', 3:'green', 4:'orange', 5:'purple', 6:'brown', 7:'pink', 8:'black', 9:'black', 10: 'brown'}
    img.save(f"{name}.png")
    draw = ImageDraw.Draw(img)
    for k,e in enumerate(anno):
        print(e)
        ## xmin , ymin , width, height
        if direct_box :
            rectangle_coords = e
            border_color = 'red' 
        else:
            rectangle_coords = e['bbox']    
            border_color = color_indicator[e['category_id']]
        if labels:
            text = labels[k]
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
            # img.save(f"temp-{k}.png")
            if text:
                position = rectangle_coords[0] - border_width , rectangle_coords[1] - border_width
                draw.text(position, text, fill='white')
    img.save(f"{name}2.png")




def set_seeds(seed, deterministic=False, ):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_random_seed(length=8):
    return random.randint(0, 10**(length) - 1)