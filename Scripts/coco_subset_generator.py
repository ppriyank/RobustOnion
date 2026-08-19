import os 
import json
from pycocotools.coco import COCO
import random 
import keyboard

def _laod_images(file):
    if not os.path.exists(file):
        return set(), set()
    coco_api = COCO(file)
    IDS = list(coco_api.imgs.keys())
    images = set()
    images_ids = set()
    for e in IDS:
        img = coco_api.loadImgs(e)[0]
        images.add(img['file_name'])
        images_ids.add(img['id'])
    return images, images_ids


def dump_new_csv(sampled_ids, file, dest):
    if os.path.exists(dest):
        pressed_key = input("Enter 'c' to continue ")
        if pressed_key != "c":quit()
    coco_api = COCO(file)
    selected_images = [img for img in coco_api.dataset['images'] if img['id'] in sampled_ids]
    selected_annotations = [ann for ann in coco_api.dataset['annotations'] if ann['image_id'] in sampled_ids]

    # Create a new dataset dictionary
    filtered_coco_dict = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco_api.dataset['categories'],
        "info": coco_api.dataset['info'],
        'licenses': coco_api.dataset['licenses'],

    }
    # Dump the filtered dictionary to a JSON file
    with open(dest, 'w') as json_file:
        json.dump(filtered_coco_dict, json_file)





root  = "/home/priyank/robustness_object_detection/"

val_json = "DATASET/instances_val2017.json"
debug_json = "DATASET/instances_val2017_debug.json"


###### val 
val_file = os.path.join(root, val_json)
val, val_ids = _laod_images(val_file)
count = len(val) 
print(f"VAL       :       {count}")
no_of_files = count 



###### Debug   
debug_file = os.path.join(root, debug_json)
debug_data, debug_ids = _laod_images(debug_file)
count = len(debug_data) 
dest = debug_file
print(f"Debug     :       {count} / {no_of_files}")

if count == 0 :
    count = 100
    print(f"Debug     :       {count} / {no_of_files}")
    sampled_ids = random.sample(val_ids, count)
    dump_new_csv(sampled_ids, val_file, dest)




# conda activate GLIP 
# python ~/robustness_object_detection/Scripts/coco_subset_generator.py

