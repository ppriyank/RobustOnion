import torch 
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np 

def save_img(x, name="temp.png", batch_dim=True, **kwargs ):
    if batch_dim:
        save_image( normalize( x[:, torch.tensor((2, 1, 0))] ), name, **kwargs )
    else:
        save_image( normalize( x[torch.tensor((2, 1, 0))] ), name, **kwargs )

def normalize(x):return (x - x.min()) / (x.max() - x.min())

def display_box_over_img(img, anno, border_width = 5, border_color = 'red', prefix_name="temp", categories=None, font=20, mode="single", font_size=None, grid_shape=None):
    save_img(img, f"{prefix_name}_OG.png")
    drawn_images = []
    for i,image in enumerate(img):
        image = to_pil_image(normalize(image))
        # image.save(f"{prefix_name}_{i}_raw.png")
        draw = ImageDraw.Draw(image)
        boxes = anno[i].bbox
        if categories is not None :
            if font_size:
                dim = font_size
            else:
                dim = max(image.size)
                dim = dim // font
            labels = anno[i].extra_fields['labels']
            if type(categories) != dict:
                categories = {e['id']:e['name'] for e in categories}
            FONT_FAMILY = "../Scripts/Arial.ttf"
            font_type = ImageFont.truetype(FONT_FAMILY, dim)

        assert anno[i].mode == "xyxy"
        for count, rectangle_coords in enumerate(boxes):
            ## xmin , ymin , xmax, ymax
            for k in range(border_width):
                draw.rectangle( [rectangle_coords[0].item() - k, rectangle_coords[1].item() - k, rectangle_coords[2].item() + k, rectangle_coords[3].item() + k], outline=border_color )
            if categories is not None:
                x = ((rectangle_coords[0] + rectangle_coords[2]) / 2).item()
                y = ((rectangle_coords[1] + rectangle_coords[3]) / 2).item()
                position = (x, y)
                text = categories[labels[count].item()]
                draw.text(position, text, fill=border_color, font=font_type)
        
        image = torch.tensor(np.array(image)).permute(2,0,1)
        if mode == "single":
            save_img(image , f"{prefix_name}_{i}.png", batch_dim=False )
        # image.convert('RGB').save(f"{prefix_name}_{i}.png")
        drawn_images.append(image)
    
    drawn_images = torch.stack(drawn_images)
    if grid_shape:
        save_img(drawn_images, f"{prefix_name}_Drawn.png", nrow=grid_shape,)
    else:
        save_img(drawn_images, f"{prefix_name}_Drawn.png")
        
        


