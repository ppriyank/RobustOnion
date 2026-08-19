import os
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import random 

# https://www.kaggle.com/datasets/shuvoalok/dawn-dataset
parser = argparse.ArgumentParser(description='dawn2coco')
parser.add_argument('--dawn_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()





os.makedirs(os.path.join(cfg.dawn_dir, 'labels_coco'), exist_ok=True)
dest_ann = os.path.join(cfg.dawn_dir, 'labels_coco', 'dawn_labels.json')

def display_box_over_img(img, anno, border_width = 5, border_color = 'red', mode="xywh"):
    img.save("temp.png")
    draw = ImageDraw.Draw(img)
    for e in anno:
        print(e)
        rectangle_coords = e
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
    img.save("temp2.png")

      


def main():
  attr_dict = {"categories":
      [
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "bicycle"},
        {"supercategory": "none", "id": 4, "name": "motorcycle"},
        {"supercategory": "none", "id": 8, "name": "truck"},
        {"supercategory": "none", "id": 6, "name": "bus"},
      ]}

  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
  id_dict_reverse = {i['id'] : i['name']  for i in attr_dict['categories']}

  images = list()
  annotations = list()
  ignore_categories = set()
  categories_observed = set()
  
  counter = 0
  local_counter = 0 
  for weather in os.listdir(cfg.dawn_dir):
      weather_path = os.path.join(cfg.dawn_dir, weather)
      
      for img in os.listdir(weather_path):
          if ".jpg" not in img:
            continue
          
          image_name = img 
          ann = os.path.join(f"{weather}_PASCAL_VOC", img.replace(".jpg", ".xml"))
          img = os.path.join(weather, img)
          img_path = os.path.join(cfg.dawn_dir, img)
      
          image = dict()
          image['file_name'] = img

          img = Image.open(img_path)
          width, height = img.size

          image['height'] = height
          image['width'] = width
          image['id'] = counter
          image['weather'] = weather
          
          ann = os.path.join(weather_path, ann)
          empty_image = True
          tmp = 0
          
          tree = ET.parse(ann)
          root = tree.getroot()
          
          assert root.findall('filename')[0].text.replace(".jpg","") == image_name.replace(".jpg",""), f"Filename : {image_name} with annotation {root.findall('filename')[0].text}"
          size = root.findall('size')[0]
          anno_w = int(size.find('width').text)
          anno_h = int(size.find('height').text)
          assert anno_w == width, "True W: {width} vs Anno W : {anno_w}"
          assert anno_h == height, "True W: {height} vs Anno W : {anno_h}"

          local_annotations = []
          
          for boxes in root.findall('object'):
              category = boxes.findall('name')[0].text
              annotation = dict()      
              categories_observed.add(category)
              box = boxes.findall('bndbox')
              assert len(box) == 1
              box = box[0]
              
              x1 = int(box.findall('xmin')[0].text)
              y1 = int(box.findall('ymin')[0].text)
              x2 = int(box.findall('xmax')[0].text)
              y2 = int(box.findall('ymax')[0].text)

              if category in id_dict.keys():
                  tmp = 1
                  empty_image = False
                  annotation["iscrowd"] = 0
                  annotation["image_id"] = counter
                  annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
                  annotation['area'] = float((x2 - x1) * (y2 - y1))
                  annotation['category_id'] = id_dict[category]
                  annotation['ignore'] = 0
                  annotation['id'] = local_counter
                  local_counter += 1 
                  annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                  annotations.append(annotation)
                  local_annotations.append(annotation['bbox'])
              else:
                ignore_categories.add(category)
                print('Ignored Category', category, img_path)
                  # /data/priyank/synthetic/Dawn_kaggle/images/snow_storm-307.jpg
              

          if random .random() > 0.997:
            display_box_over_img(img, local_annotations, border_width = 5, border_color = 'red', mode="xywh")

          if empty_image:
            print('empty image!', ann)
            continue
          if tmp == 1:
            images.append(image)
            counter += 1
      
  
  print( " TOTAL NO OF IMAGES :: ", len(images) )
  #  TOTAL NO OF IMAGES ::  1026
  print( " TOTAL NO OF BOXES :: ", len(annotations) )
  #  TOTAL NO OF BOXES ::  7845

  
  attr_dict["images"] = images
  attr_dict["annotations"] = annotations
  attr_dict["type"] = "instances"

  print('ignored categories: ', ignore_categories)
  print('saving...')
  with open(dest_ann, "w") as file:
    json.dump(attr_dict, file)
  print('Done.')











if __name__ == '__main__':
  main()


# cd ~/robustness_object_detection/
# python Scripts/dawn2coco.py --dawn_dir /data/priyank/synthetic/DAWN2