import os
import json
import argparse
from tqdm import tqdm



parser = argparse.ArgumentParser(description='bdd2coco')
parser.add_argument('--bdd_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_val.json')
src_train_dir = os.path.join(cfg.bdd_dir, 'labels', 'bdd100k_labels_images_train.json')

save_dir = "DATASET/"
save_dir_train = os.path.join(save_dir, 'bdd100k_labels_images_train_ANNO.json') 
save_dir_val = os.path.join(save_dir, 'bdd100k_labels_images_val_ANNO.json') 


def bdd2coco_detection(labeled_images, save_dir):
  attr_dict = {}
  annotations = list()
  images = list()
  counter = 0
  weathers = set()
  day_time = set()
  for i in tqdm(labeled_images):
    weather = i['attributes']['weather']
    time_local = i['attributes']['timeofday']
    day_time.add(time_local)
    weathers.add(weather)

    counter += 1
    assert i['name'] not in attr_dict
    attr_dict[i['name']] = i['attributes']
    
  print('saving...')
  with open(save_dir, "w") as file:
    json.dump(attr_dict, file)
  print('Done.')
  print(f"Weathers :: {weathers}")
  print(f"Day Time :: {day_time}")

def main():
  # create BDD training set detections in COCO format
  print('Loading training set...')
  with open(src_train_dir) as f:
    train_labels = json.load(f)
  print('Converting training set...')
  bdd2coco_detection(train_labels, save_dir_train)

  # create BDD validation set detections in COCO format
  print('Loading validation set...')
  with open(src_val_dir) as f:
    val_labels = json.load(f)
  print('Converting validation set...')
  bdd2coco_detection(val_labels, save_dir_val)


if __name__ == '__main__':
  main()


# cd ~/robustness_object_detection/
# python Scripts/bdd_weather_labels.py --bdd_dir /data/priyank/synthetic/bdd100k/


# Weathers :: {'partly cloudy', 'snowy', 'rainy', 'foggy', 'overcast', 'undefined', 'clear'}
# Day Time :: {'undefined', 'night', 'dawn/dusk', 'daytime'}

# Weathers :: {'partly cloudy', 'snowy', 'rainy', 'foggy', 'overcast', 'undefined', 'clear'}
# Day Time :: {'undefined', 'night', 'dawn/dusk', 'daytime'}