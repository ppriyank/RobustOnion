
import os
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import random 
from collections import defaultdict

# https://www.kaggle.com/datasets/shuvoalok/dawn-dataset
parser = argparse.ArgumentParser(description='dawn2coco')
parser.add_argument('--Cityscape', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()




ann_root = os.path.join(cfg.Cityscape, 'gtBbox3d')
os.makedirs(os.path.join(cfg.Cityscape, 'labels_coco'), exist_ok=True)


def display_box_over_img(img, anno, border_width = 5, border_color = 'red', mode="xywh", output="temp2.png"):
    img.save("temp.png")
    draw = ImageDraw.Draw(img)
    for e in anno:
        print(e)
        rectangle_coords = e
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
    img.save(output)

      


def main():
  # {'bus', 'bicycle', 'motorcycle', 'trailer', 'truck', 'car'}
  #### Train 
  # defaultdict(<class 'int'>, {'car': 18447, 'bicycle': 2744, 'bus': 346, 'motorcycle': 528, 'truck': 327, 'trailer': 52, 'tunnel': 1, 'train': 273, 'dynamic': 4})
  #### VAL 
  # val  ::  defaultdict(<class 'int'>, {'car': 3049, 'bus': 104, 'bicycle': 881, 'motorcycle': 102, 'trailer': 7, 'truck': 63, 'train': 41, 'caravan': 2})
  attr_dict = {"categories":
      [
        {"supercategory": "none", "id": 1, "name": "car"},
        {"supercategory": "none", "id": 2, "name": "bicycle"},
        {"supercategory": "none", "id": 3, "name": "bus"},
        {"supercategory": "none", "id": 4, "name": "motorcycle"},
        {"supercategory": "none", "id": 5, "name": "truck"},
      ]}

  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
  id_dict_reverse = {i['id'] : i['name']  for i in attr_dict['categories']}

  
  weather = 'fog'
  box_type = 'amodal'
  # box_type = 'modal'
  for folder in  ['train', 'val']:
    counter = 0
    local_counter = 0 
    images = list()
    annotations = list()
    ignore_categories = set()
    categories_observed = defaultdict(int)
    dest_ann = os.path.join(cfg.Cityscape, 'labels_coco', f'Cityscape_labels_{folder}_{box_type}.json')
    imgage_folder = os.path.join(cfg.Cityscape, folder)
    for city in os.listdir(imgage_folder):
        city_path = os.path.join(imgage_folder, city)
        ann_root_local = os.path.join(ann_root, folder, city)
        done = set()
        for img in os.listdir(city_path):
            if ".png" not in img:
              continue
            file_name = img.split('_leftImg8bit_foggy_')[0]
            if file_name in done:
              continue 
            done.add(file_name)

            ann = os.path.join(ann_root_local, file_name + '_gtBbox3d.json')
            
            # 'ulm_000003_000019_leftImg8bit_foggy_beta_0.005.png'
            # ulm_000019_000019_gtBbox3d.json

            image_name = img 
            img_path = os.path.join(city_path, img)
        
            image = dict()
            image['file_name'] = os.path.join(city, file_name)

            img = Image.open(img_path)
            width, height = img.size

            image['height'] = height
            image['width'] = width
            image['id'] = counter
            image['weather'] = weather
            
            
            empty_image = True
            tmp = 0
            
            with open(ann, 'r') as file:
              # Use json.load() to parse the JSON data from the file
              # and convert it into a Python dictionary or list
              ann = json.load(file )

            anno_w = ann['imgWidth']
            anno_h=  ann['imgHeight']
            assert anno_w == width, "True W: {width} vs Anno W : {anno_w}"
            assert anno_h == height, "True W: {height} vs Anno W : {anno_h}"

            
            local_annotations = []
            for boxes in ann['objects']:
                category = boxes ['label']
                categories_observed[category] += 1
                
                box = boxes['2d'][box_type]
                annotation = dict()      
                
                # x_min, y_min, Width, Height
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if category in id_dict.keys():
                    tmp = 1
                    empty_image = False
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = counter
                    annotation['bbox'] = [x1, y1, x2, y2]
                    annotation['area'] = float((x2) * (y2))
                    annotation['category_id'] = id_dict[category]
                    annotation['ignore'] = 0
                    annotation['id'] = local_counter
                    local_counter += 1 
                    annotation['segmentation'] = []
                    annotations.append(annotation)
                    local_annotations.append(annotation['bbox'])
                else:
                  ignore_categories.add(category)
                  print('Ignored Category', category, img_path)
            
            if random .random() > 0.997:
              display_box_over_img(img, local_annotations, mode="xywh")

            if empty_image:
              print('empty image!', ann)
              continue
            if tmp == 1:
              images.append(image)
              counter += 1

            
    print(folder , " :: ", categories_observed)
    # train  ::  defaultdict(<class 'int'>, {'car': 18447, 'bicycle': 2744, 'bus': 346, 'motorcycle': 528, 'truck': 327, 'trailer': 52, 'tunnel': 1, 'train': 273, 'dynamic': 4})
    # val  ::  defaultdict(<class 'int'>, {'car': 3049, 'bus': 104, 'bicycle': 881, 'motorcycle': 102, 'trailer': 7, 'truck': 63, 'train': 41, 'caravan': 2})

    print(f"{folder} ::  TOTAL NO OF IMAGES :: ", len(images) )
    # train ::  TOTAL NO OF IMAGES ::  2908
    # val ::  TOTAL NO OF IMAGES ::  488

    print(f"{folder} ::  TOTAL NO OF BOXES :: ", len(annotations) )
    # train ::  TOTAL NO OF BOXES ::  22392
    # val ::  TOTAL NO OF BOXES ::  4199

    print(f"{folder} ::  Ignored Categories :: ", ignore_categories )
    # train ::  Ignored Categories ::  {'tunnel', 'dynamic', 'train', 'trailer'}
    # val ::  Ignored Categories ::  {'caravan', 'train', 'trailer'}

    

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    
    print('saving...')
    with open(dest_ann, "w") as file:
      json.dump(attr_dict, file)
    print('Done.')







if __name__ == '__main__':
  main()
  # check_specific("dust_16.jpg")
  # check_specific("summer_179.jpg")





# cd ~/robustness_object_detection/
# python Scripts/FoggyCityscape2coco.py --Cityscape /data/priyank/synthetic/leftImg8bit_foggyDBF 
