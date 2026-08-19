import os
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import random 
import pandas as pd 



# https://www.kaggle.com/datasets/shuvoalok/dawn-dataset
parser = argparse.ArgumentParser(description='VisDrone20192coco')
parser.add_argument('--visiondrone_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

os.makedirs(os.path.join(cfg.visiondrone_dir, 'labels_coco'), exist_ok=True)
dest_val_ann = os.path.join(cfg.visiondrone_dir, 'labels_coco', 'visiondrone_val_label.json')



def display_box_over_img(img, anno, border_width = 5, border_color = 'red', mode="xywh"):
    img.save("temp.png")
    draw = ImageDraw.Draw(img)
    for e in anno:
        # print(e)
        rectangle_coords = e
        if mode == "xywh":
            rectangle_coords =rectangle_coords[0], rectangle_coords[1], rectangle_coords[0] + rectangle_coords[2], rectangle_coords[1] + rectangle_coords[3]
        for i in range(border_width):
            draw.rectangle( [rectangle_coords[0] - i, rectangle_coords[1] - i, rectangle_coords[2] + i, rectangle_coords[3] + i], outline=border_color )
    img.save("temp2.png")


#  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
#     Name                                                  Description
# -------------------------------------------------------------------------------------------------------------------------------     
#  <bbox_left>	     The x coordinate of the top-left corner of the predicted bounding box

#  <bbox_top>	     The y coordinate of the top-left corner of the predicted object bounding box

#  <bbox_width>	     The width in pixels of the predicted object bounding box

# <bbox_height>	     The height in pixels of the predicted object bounding box

#    <score>	     The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
#                      an object instance.
#                      The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
#                      while 0 indicates the bounding box will be ignored.
                      
# <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
#                      people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
#                      others(11))
                      
# <truncation>	     The score in the DETECTION result file should be set to the constant -1.
#                      The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
#                      (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
                      
# <occlusion>	     The score in the DETECTION file should be set to the constant -1.
#                      The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
#                      (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
#                      (occlusion ratio 50% ~ 100%)).


def main(easy=None):
    attr_dict = {"categories":
        [
        # {"supercategory": "none", "id": 1, "name": "pedestrian"},
        {"supercategory": "none", "id": 2, "name": "person"},
        {"supercategory": "none", "id": 3, "name": "bicycle"},
        {"supercategory": "none", "id": 4, "name": "car"},
        {"supercategory": "none", "id": 5, "name": "van"},
        {"supercategory": "none", "id": 6, "name": "truck"},
        # {"supercategory": "none", "id": 7, "name": "tricycle"},
        {"supercategory": "none", "id": 9, "name": "bus"},
        {"supercategory": "none", "id": 10, "name": "motorcycle"},
        # {"supercategory": "none", "id": 11, "name": "awning-tricycle"},
        ]}
    
    # rename pedestrian to person 
    class_renamer = {1 :  2}
    
    # tricycle and awning-tricycle too difficult 
    # awning-tricycle(8), 
    # (11),
    # pedestrian (walking) is same as person (riding) something
    #  {"supercategory": "none", "id": 1, "name": "pedestrian"},
    # {"supercategory": "none", "id": 2, "name": "person"},
    id_dict_reverse = {i['id'] : i['name']  for i in attr_dict['categories']}
    id_dict_reverse.update({
        8: 'awning-tricycle', 11:'others', 0:'ignroe', 7: 'tricycle',
    })
    
    if easy:
        attr_dict = {"categories":
        [
        {"supercategory": "none", "id": 4, "name": "car"},
        {"supercategory": "none", "id": 6, "name": "truck"},
        {"supercategory": "none", "id": 9, "name": "bus"},
        ]}
        id_dict_reverse.update({
            8: 'awning-tricycle', 11:'others', 0:'ignroe', 7: 'tricycle',
            1: "pedestrian", 2: "person", 3: "bicycle", 5: "van", 10: "motorcycle", 
        })
        class_renamer = {}
    
    id_dict = {i['name']: i['id'] for i in attr_dict['categories']}


    images = list()
    annotations = list()
    ignore_categories = set()
    categories_observed = set()
    counter = 0
    local_counter = 0 
    
    
    for ann_file in os.listdir( os.path.join(cfg.visiondrone_dir, 'annotations') ):
        image_name = ann_file.replace(".txt" , "") + ".jpg"
        img_path = os.path.join(cfg.visiondrone_dir, 'images', image_name)  
        image = dict()
        image['file_name'] = image_name

        img = Image.open(img_path)
        width, height = img.size

        image['height'] = height
        image['width'] = width
        image['id'] = counter
        
        empty_image = True
        tmp = 0    
        ann_file = os.path.join(cfg.visiondrone_dir, 'annotations', ann_file)
        df = pd.read_csv(ann_file, names=[
            'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion'])
        
        df = df[df.score == 1] 
        # print(df)
        local_annotations = []
        for index, box in df.iterrows():
            category = box.object_category
            if category in class_renamer:
                category = class_renamer [category]
            
            category = id_dict_reverse[ category ]
            annotation = dict()      
            categories_observed.add(category)

            
            x1 = int(box.bbox_left)
            y1 = int(box.bbox_top)
            w = int(box.bbox_width)
            h = int(box.bbox_height)
            
            if category in id_dict.keys():
                tmp = 1
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = counter
                annotation['bbox'] = [x1, y1, w, h]
                annotation['area'] = float( w * h)
                annotation['category_id'] = id_dict[category]
                annotation['ignore'] = 0
                annotation['id'] = local_counter
                local_counter += 1 
                # annotation['segmentation'] = []
                annotation['segmentation'] = [[x1, y1, x1, y1+w, x1 + w, y1+w, x1 + w, y1]]
                annotations.append(annotation)
                local_annotations.append(annotation['bbox'])
            else:
                ignore_categories.add(category)
                # print('Ignored Category', category, img_path)

               
        if random .random() > 0.997:
            display_box_over_img(img, local_annotations, border_width = 2, border_color = 'red', mode="xywh")

        if empty_image:
            print('empty image!', box)
            continue
        if tmp == 1:
            images.append(image)
            counter += 1
      
    
    print( " TOTAL NO OF IMAGES :: ", len(images) )
    #  TOTAL NO OF IMAGES ::  548
    # EASY :: TOTAL NO OF IMAGES ::  518 
    print( " TOTAL NO OF BOXES :: ", len(annotations) )
    #  TOTAL NO OF BOXES ::  38227
    # EASY :: TOTAL NO OF IMAGES ::  15065

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('ignored categories: ', ignore_categories)
    # ignored categories:  {'awning-tricycle'}
    print('saving...')
    if easy:
        global dest_val_ann
        dest_val_ann = dest_val_ann.replace('.json', '-easy.json')
    with open(dest_val_ann, "w") as file:
        json.dump(attr_dict, file)
    print('Done.')











if __name__ == '__main__':
  main(easy=True)


# cd ~/robustness_object_detection/
# python Scripts/VisDrone20192coco.py --visiondrone_dir /data/priyank/synthetic/VisDrone2019-DET-val/