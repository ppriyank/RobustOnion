import os
import json
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import random 
import pandas as pd 



# https://www.kaggle.com/datasets/shuvoalok/dawn-dataset
parser = argparse.ArgumentParser(description='UAV-benchmark-MOTD_v1.0')
parser.add_argument('--UAVDT_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

os.makedirs(os.path.join(cfg.UAVDT_dir, 'labels_coco'), exist_ok=True)
dest_val_ann = os.path.join(cfg.UAVDT_dir, 'labels_coco', 'UAVDT_train_label.json')



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


# DET Groundtruth Format (*_gt_whole.txt)
# It looks as follows:
#         <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<out-of-view>,<occlusion>,<object_category>
#      -----------------------------------------------------------------------------------------------------------------------------------
#            Name	                                      Description
#      -----------------------------------------------------------------------------------------------------------------------------------
#        <frame_index>	  The frame index of the video frame
#         <target_id>	  In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding 
# 			          relation of the bounding boxes in different frames.  
#         <bbox_left>	          The x coordinate of the top-left corner of the predicted bounding box
#         <bbox_top>	          The y coordinate of the top-left corner of the predicted object bounding box
#         <bbox_width>	  The width in pixels of the predicted object bounding box
#         <bbox_height>	  The height in pixels of the predicted object bounding box
#         <out-of-view>	     The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
# 			          (i.e., 'no-out'= 1,'medium-out' =2,'small-out'=3).
#          <occlusion>	  The score in the GROUNDTRUTH fileindicates the fraction of objects being occluded.
#                         (i.e.,'no-occ'=1,'lagre-occ'=2,'medium-occ'=3,'small-occ'=4).
#      <object_category>	  The object category indicates the type of annotated object, (i.e.,car(1), truck(2), bus(3))


def main():
    attr_dict = {"categories":
        [
            {"supercategory": "none", "id": 1, "name": "car"},
            {"supercategory": "none", "id": 2, "name": "truck"},
            {"supercategory": "none", "id": 3, "name": "bus"},
        ]}
   
    id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
    id_dict_reverse = {i['id'] : i['name']  for i in attr_dict['categories']}
    
    images = list()
    annotations = list()
    ignore_categories = set()
    categories_observed = set()
    counter = 0
    local_counter = 0 
    
    for ann_file in os.listdir( os.path.join(cfg.UAVDT_dir, 'UAV-benchmark-MOTD_v1.0', 'GT') ):
        if '_gt_whole.txt' not in ann_file:
            continue 
        image_name_folder = ann_file.replace("_gt_whole.txt" , "")
        
        ann_file = os.path.join(cfg.UAVDT_dir, 'UAV-benchmark-MOTD_v1.0', 'GT', ann_file)

        df = pd.read_csv(ann_file)
        df = pd.read_csv(ann_file, names=[
            'frame_no', 'target_id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 
            'out-of-view', 'occlusion', 'object_category'])
        
        # print(df.object_category.unique(),  df['out-of-view'].unique(), df['occlusion'].unique())
        
        for frame in df.frame_no.unique():
            image_name = os.path.join(image_name_folder, f'img{frame:>06}.jpg')
            local_df = df[df.frame_no == frame]
            img_path = os.path.join(cfg.UAVDT_dir, 'UAV-benchmark-M', image_name)  
            image = dict()
            image['file_name'] = image_name

            img = Image.open(img_path)
            width, height = img.size

            image['height'] = height
            image['width'] = width
            image['id'] = counter
        
            empty_image = True
            tmp = 0    
            
            local_annotations = []
            for index, box in local_df.iterrows():
                category = id_dict_reverse[box.object_category]
                annotation = dict()      
                categories_observed.add(category)
                
                if category in id_dict.keys():
                    tmp = 1
                    empty_image = False
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = counter
                    annotation['bbox'] = [int(box.bbox_left), int(box.bbox_top), int(box.bbox_width), int(box.bbox_height)]
                    annotation['area'] = float( box.bbox_width * box.bbox_height)
                    annotation['category_id'] = id_dict[category]
                    annotation['ignore'] = 0
                    annotation['id'] = local_counter
                    local_counter += 1 
                    annotation['segmentation'] = []
                    annotations.append(annotation)
                    local_annotations.append(annotation['bbox'])
                else:
                    ignore_categories.add(category)
                    # print('Ignored Category', category, img_path)

            if random .random() > 0.997:
                display_box_over_img(img, local_annotations, border_width = 2, border_color = 'red', mode="xywh")

            if empty_image:
                print('empty image!', ann)
                continue
            if tmp == 1:
                images.append(image)
                counter += 1
      
    
    print( " TOTAL NO OF IMAGES :: ", len(images) )
    #  TOTAL NO OF IMAGES ::  40409
    print( " TOTAL NO OF BOXES :: ", len(annotations) )
    #  TOTAL NO OF BOXES ::  798795

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('ignored categories: ', ignore_categories)
    # ignored categories:  {'awning-tricycle'}
    print('saving...')
    with open(dest_val_ann, "w") as file:
        json.dump(attr_dict, file)
    print('Done.')











if __name__ == '__main__':
  main()


# cd ~/robustness_object_detection/
# python Scripts/UAV-benchmark2coco.py --UAVDT_dir /data/priyank/synthetic/UAVDT/
