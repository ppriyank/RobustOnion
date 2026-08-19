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
    }
    # Dump the filtered dictionary to a JSON file
    with open(dest, 'w') as json_file:
        json.dump(filtered_coco_dict, json_file)





root = "/home/c3-0/datasets/flickr_dataset_30k/"
all_imgs = "/home/c3-0/datasets/flickr_dataset_30k/flickr30k/flickr30k-images"

root = "/data/priyank/synthetic/flickr_dataset_30k/"
all_imgs = "/data/priyank/synthetic/flickr_dataset_30k/flickr30k/flickr30k-images"

train_json = "mdetr_annotations/final_flickr_separateGT_train.json"
val_json = "mdetr_annotations/final_flickr_separateGT_val.json"
test_json = "mdetr_annotations/final_flickr_separateGT_test.json"


sampled_json = "mdetr_annotations/final_flickr_separateGT_train_subset.json"
debug_json = "DATASET/final_flickr_separateGT_train_debug.json"
sampled_json2 = "DATASET/final_flickr_separateGT_train_subset2.json"
sampled_json3 = "DATASET/final_flickr_separateGT_train_subset3.json"
sampled_json4 = "DATASET/final_flickr_separateGT_train_subset4.json"


no_of_files = len(os.listdir(all_imgs)) - 1 
all_files = set(os.listdir(all_imgs))

# FILE='/home/c3-0/datasets/flickr_dataset_30k/mdetr_annotations/final_flickr_separateGT_train_subset.json'
# rsync -a ucf0:$FILE ~/VLM-LR/glip/GLIP/


###### train 
file = os.path.join(root, train_json)
train, train_ids = _laod_images(file)
count = len(train) 
print(f"Train       :       {count} / {no_of_files}")

###### Val
file = os.path.join(root, val_json)
val, val_ids = _laod_images(file)
count = len(val) 
print(f"VAL         :       {count} / {no_of_files}")

###### test
file = os.path.join(root, test_json)
test, test_ids = _laod_images(file)
count = len(test) 
print(f"Test        :       {count} / {no_of_files}")


###### TOTAL & SANITY CHECK 
count  = len(test) + len(val) + len(train)
print(f"Total       :       {count} / {no_of_files}")

assert len(train & val) == 0
assert len(train & test) == 0
assert len(val & test) == 0

Remaining = all_files.difference(train).difference(val).difference(test)
print("Unassigned  : " , Remaining)



# ###### Debug   
# train_file = os.path.join('DATASET/final_flickr_separateGT_train_subset3.json')
# train, train_ids = _laod_images(train_file)
# file = debug_json
# sample, sample_ids = _laod_images(file)
# count = len(sample) 
# dest = debug_json

# print(f"Debug        :       {count} / {no_of_files}")
# if count == 0 :
#     count = 20
#     print(f"Debug       :       {count} / {len(train)}")
#     sampled_ids = random.sample(train_ids, count)
#     dump_new_csv(sampled_ids, train_file, dest)



# ###### TRAIN SAMPLE 
# train_ids = list(train_ids)
# train_file = os.path.join(root, train_json)
# dest = os.path.join(root, sampled_json)

# ###### Sampling  
# file = os.path.join(root, sampled_json)
# sample, sample_ids = _laod_images(file)
# count = len(sample) 
# print(f"Sample      :       {count} / {no_of_files}")

# if count == 0 or count == 1000 :
#     count = (len(train) * 0.1)
#     count = int(count)
#     print(f"Subset     :       {count} / {len(train)}")
#     sampled_ids = random.sample(train_ids, count)
#     dump_new_csv(sampled_ids, train_file, dest)



# ###### Subset 2    
# file = sampled_json2
# sample, sample_ids = _laod_images(file)
# count = len(sample) 
# dest = sampled_json2

# print(f"Subset2       :       {count} / {no_of_files}")
# if count == 0 :
#     count = (len(train) * 0.2) // 1
#     count = int(count)
#     print(f"Subset2       :       {count} / {len(train)}")
#     sampled_ids = random.sample(train_ids, count)
#     dump_new_csv(sampled_ids, train_file, dest)

###### Subset 3    
file = sampled_json3
sample, sample_ids = _laod_images(file)
count = len(sample) 
dest = sampled_json3

print(f"Subset3       :       {count} / {no_of_files}")
if count == 0 :
    count = 10000
    print(f"Subset3       :       {count} / {len(train)}")
    sampled_ids = random.sample(train_ids, count)
    dump_new_csv(sampled_ids, train_file, dest)

###### Subset 4   
train_file = os.path.join(root, train_json)
file = sampled_json4
sample, sample_ids = _laod_images(file)
count = len(sample) 
dest = sampled_json4

print(f"Subset4       :       {count} / {no_of_files}")
if count == 0 :
    count = 5000
    print(f"Subset4       :       {count} / {len(train)}")
    sampled_ids = random.sample(train_ids, count)
    dump_new_csv(sampled_ids, train_file, dest)








# conda activate GLIP 
# cd ~/robustness_object_detection/
# python Scripts/flickr_subset_generator.py

