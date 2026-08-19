import os
import json
import argparse
from tqdm import tqdm
import pandas as pd 
from PIL import Image, ImageDraw, ImageFont
import random 

parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--kitti', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.kitti, 'vkitti_2.0.3_textgt', )
src_train_dir = os.path.join(cfg.kitti, 'vkitti_2.0.3_textgt', )

label_folder=  os.path.join(cfg.kitti, 'labels_coco')

try:
    os.mkdir(label_folder)
except FileExistsError:
  _  = 0 
    
dst_val_dir = os.path.join(label_folder, 'kitti_labels_images_val_coco.json')
dst_train_dir = os.path.join(label_folder, 'kitti_labels_images_train_coco.json')



def display_box_over_img(image_path, anno, border_width = 5, mode="xywh", direct_box=False, root=None, labels=None, ):
    
    color_indicator = {0: 'red', 1:'blue', 2:'yellow', 3:'green'}
    # image_path = os.path.join(root, image_path)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    for k,e in enumerate(anno):
        # print(e)
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


def kitti2coco_detection(scenes, save_dir, mode='train', display_sample=False, global_class=None, classes=None , rev_class=None ):
  # left (x-coordinate of the left edge, which is xmin)
  # top (y-coordinate of the top edge, which is ymin)
  # right (x-coordinate of the right edge, which is xmax)
  # bottom (y-coordinate of the bottom edge, which is ymax)
  
  attr_dict = {"categories":
    [
      {"supercategory": "none", "id": 1, "name": "car"},
      {"supercategory": "none", "id": 2, "name": "truck"},
      {"supercategory": "none", "id": 3, "name": "van"},
    ]
  }
  id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
  id_counter = {i['name']: 0 for i in attr_dict['categories']}

  images = list()
  annotations = list()
  ignore_categories = set()
  counter = 0
  local_id = 0 
  print('Converting training set...')
  for scene in scenes:
      scene_path = os.path.join(src_train_dir, scene)
      for condition in os.listdir(scene_path):
        condition_path = os.path.join(scene_path, condition)
        
        label_path = os.path.join(condition_path, 'bbox.txt')
        df = pd.read_csv(label_path, sep=' ')

        frames = df.frame.unique()
        rgb_src = os.path.join(scene, condition, 'frames', 'rgb')

        box_id_path = os.path.join(condition_path, 'info.txt')
        box_id = pd.read_csv(box_id_path, sep=' ')

        df = pd.merge(df, box_id, on='trackID', how='inner')

        for frame in tqdm(frames):
          for camera in [0,1]:
            counter += 1
            
            relative_path = os.path.join(rgb_src, f'Camera_{camera}', f'rgb_{frame:05d}.jpg')
            rgb_path = os.path.join(cfg.kitti, relative_path)
            assert os.path.exists(rgb_path)
            img = Image.open(rgb_path)
            
            image = dict()
            image['file_name'] = relative_path
            image['height'] = img.size[1]
            image['width'] = img.size[0]
            image['id'] = counter
            image['condition'] = condition
            image['scene'] = scene

            empty_image = True

            tmp = 0
            sample = df[ (df.frame  == frame) & (df.cameraID == camera)]
            if len(sample) ==0 :
              continue 
            anno = [(row.left, row.top, row.right, row.bottom) for index, row in sample.iterrows()]

            labels_string = sample.label.tolist()
            labels = [classes[e] for e in labels_string]

            areas = sample.number_pixels.tolist()
            occupancies = sample.occupancy_ratio.tolist()

            if display_sample and random.random() > 0.99 and len(set(labels)) == 3 :
                display_box_over_img(rgb_path, anno, labels=labels, border_width = 3, mode="xyxy", direct_box=True, root=None)
                

            
            for element in zip(anno, labels, areas, occupancies):
              if element[3] == 0 or element [2] == 0:
                continue
              area = int(element [2] / element[3])

              
              annotation = dict()
              tmp = 1
              empty_image = False
              annotation["iscrowd"] = 0
              annotation["image_id"] = image['id']
              x1 = element[0][0]
              y1 = element[0][1]
              x2 = element[0][2]
              y2 = element[0][3]
              annotation['bbox'] = [x1, y1, x2 - x1, y2 - y1]
              annotation['area'] = float((x2 - x1) * (y2 - y1))
              assert abs(annotation['area'] - area) < 2, f"Img {rgb_path} , Computed Area : {annotation['area']} vs Annotated {element[2]} "
              
              annotation['category_id'] = element[1]
              annotation['ignore'] = 0
              annotation['id'] = local_id
              local_id +=1
              annotation['segmentation'] = []
              annotations.append(annotation)

              id_counter[ rev_class[element[1]].lower() ] += 1
          
            if empty_image:
              print('empty image!', sample)
              continue
            
            if tmp == 1:
              images.append(image)

  attr_dict["images"] = images
  attr_dict["annotations"] = annotations
  attr_dict["type"] = "instances"

  print("Counter of categories", id_counter)
  print('saving...')
  
  with open(save_dir, "w") as file:
      json.dump(attr_dict, file)
  print('Done.')

      
      
    

def main(val_only=False):
  
  classes = {'Car': 1, 'Truck': 2,  'Van': 3, }
  rev_class=  {1: 'Car', 2: 'Truck', 3: 'Van'}
        
    
  if not val_only:
    # create Kitti training set detections in COCO format
    scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18'  ]
    print('Loading training set...' , scenes)
    kitti2coco_detection(scenes=scenes, save_dir=dst_train_dir, mode = 'train', classes=classes, rev_class=rev_class, display_sample=True)
    # Counter of categories {'car': 191913, 'truck': 5959, 'van': 11102}
    
  scenes = ['Scene20' ]
  print('Loading validation set...' , scenes)
  kitti2coco_detection(scenes=scenes, save_dir=dst_val_dir, mode = 'test', classes=classes, rev_class=rev_class, display_sample=True)
  # Counter of categories {'car': 292971, 'truck': 2849, 'van': 28054}
  

def image_stats(dump_path='Viz', dump_only_min=True):
  try:
      os.mkdir(dump_path)
  except FileExistsError:
    _  = 0 

  for scene in ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']:
    scene_path = os.path.join(cfg.kitti, scene)
    for folder in os.listdir(scene_path):
      folder_path = os.path.join(scene_path, folder, 'frames/rgb/')
      for camera in os.listdir(folder_path):
        camera_path = os.path.join(folder_path, camera)
        images = os.listdir(camera_path)
        mini, maxi = min(images), max(images)
        mini_path = os.path.join(camera_path, mini)
        maxi_path = os.path.join(camera_path, maxi)
        
        dest = os.path.join(dump_path, f"{scene}-{folder}-{camera}")
        
        os.system(f'cp {mini_path} {dest}-{mini}')
        if not dump_only_min:
          os.system(f'cp {maxi_path} {dest}-{maxi}')
        
        

  
    

if __name__ == '__main__':
  # main(val_only=False )
  # Train 
  # Scene01  Scene02  Scene06  Scene18  

  # Test
  # Scene20

  image_stats()


# cd ~/robustness_object_detection/
# python Scripts/virtual_kitt2coco.py --kitti /data/priyank/synthetic/VirtualKitti 



