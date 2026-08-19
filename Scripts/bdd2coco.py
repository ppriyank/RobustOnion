import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_train.json')

os.makedirs(os.path.join(cfg.bdd_dir, 'labels_coco'), exist_ok=True)

dst_val_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_labels_images_val_coco.json')
dst_train_dir = os.path.join(cfg.bdd_dir, 'labels_coco', 'bdd100k_labels_images_train_coco.json')



def display_box_over_img(image_path, anno, border_width = 5, mode="xywh", direct_box=False, root=None):
    from PIL import Image, ImageDraw, ImageFont
    color_indicator = {0: 'red', 1:'blue', 2:'yellow', 3:'green', 4:'orange', 5:'purple', 6:'brown', 7:'pink', 8:'black'}
    image_path = os.path.join(root, image_path)
    img = Image.open(image_path)
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
            border_color = 'red'
            if e['category_id'] in  color_indicator:
              border_color = color_indicator[e['category_id']]
            
              
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
    img.save("temp2.png")


def bdd2coco_detection(labeled_images, save_dir, mode='train',):
  attr_dict = {"categories":
    [
      {"supercategory": "none", "id": 1, "name": "person"},
      {"supercategory": "none", "id": 2, "name": "car"},
      {"supercategory": "none", "id": 3, "name": "rider"},
      {"supercategory": "none", "id": 4, "name": "bus"},
      {"supercategory": "none", "id": 5, "name": "truck"},
      {"supercategory": "none", "id": 6, "name": "bike"},
      {"supercategory": "none", "id": 7, "name": "motor"},
      {"supercategory": "none", "id": 8, "name": "traffic light"},
      {"supercategory": "none", "id": 9, "name": "traffic sign"},
      # {"supercategory": "none", "id": 10, "name": "train"},
    ]}
  renamer = {'bike' : 'bicycle',  'motor': "motorcycle"}
  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
  id_counter = {i['name']: 0 for i in attr_dict['categories']}

  
  
  images = list()
  annotations = list()
  ignore_categories = set()

  counter = 0
  for i in tqdm(labeled_images):
    counter += 1
    image = dict()
    image['file_name'] = i['name']
    image['height'] = 720
    image['width'] = 1280

    image['id'] = counter

    empty_image = True

    tmp = 0
    for l in i['labels']:
      annotation = dict()
      if l['category'] in id_dict.keys():
        tmp = 1
        empty_image = False
        annotation["iscrowd"] = 0
        annotation["image_id"] = image['id']
        x1 = l['box2d']['x1']
        y1 = l['box2d']['y1']
        x2 = l['box2d']['x2']
        y2 = l['box2d']['y2']
        annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
        annotation['area'] = float((x2 - x1) * (y2 - y1))
        annotation['category_id'] = id_dict[l['category']]
        annotation['ignore'] = 0
        annotation['id'] = l['id']
        annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        annotations.append(annotation)
        id_counter[ l['category'] ] += 1
      else:
        ignore_categories.add( l['category'] )

    if empty_image:
      print('empty image!')
      continue
    
    
    # display_box_over_img(image['file_name'], annotations, root=f'{cfg.bdd_dir}/images/100k/{mode}/' )
    # quit()

    if tmp == 1:
      images.append(image)

  attr_dict["images"] = images
  attr_dict["annotations"] = annotations
  attr_dict["type"] = "instances"

  print('ignored categories: ', ignore_categories)
  print("Counter of categories", id_counter)
  print('saving...')
  
  # attr_dict['categories'][5]['name'] = renamer[attr_dict['categories'][5]['name']]
  # attr_dict['categories'][6]['name'] = renamer[attr_dict['categories'][6]['name']]
  
  with open(save_dir, "w") as file:
    json.dump(attr_dict, file)
  print('Done.')


def main(val_only=False):
  if not val_only:
    # create BDD training set detections in COCO format
    print('Loading training set...')
    with open(src_train_dir) as f:
      train_labels = json.load(f)
    print('Converting training set...')
    bdd2coco_detection(train_labels, dst_train_dir, mode = 'train')

    # ignored categories:  {'lane', 'train', 'drivable area'}
    # Counter of categories {'person': 91349, 'car': 713211, 'rider': 4517, 'bus': 11672, 'truck': 29971, 'bike': 7210, 'motor': 3002, 'traffic light': 186117, 'traffic sign': 239686}




  # create BDD validation set detections in COCO format
  print('Loading validation set...')
  with open(src_val_dir) as f:
    val_labels = json.load(f)
  print('Converting validation set...')
  bdd2coco_detection(val_labels, dst_val_dir, mode='val')

  # ignored categories:  {'lane', 'train', 'drivable area'}
  # Counter of categories {'person': 13262, 'car': 102506, 'rider': 649, 'bus': 1597, 'truck': 4245, 'bike': 1007, 'motor': 452, 'traffic light': 26885, 'traffic sign': 34908}





if __name__ == '__main__':
  main(val_only=False )


# cd ~/robustness_object_detection/
# python Scripts/bdd2coco.py --bdd_dir /data/priyank/synthetic/bdd100k/

# rsync -a /data/priyank/synthetic/bdd100k/labels_coco/* ucf0:/home/c3-0/datasets/bdd100k/labels_coco/
# ~/robustness_object_detection/DATASET/