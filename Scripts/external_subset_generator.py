import os 
import json
from pycocotools.coco import COCO
import random 
from collections import defaultdict 

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

def _laod_images_with_categories(file, categories_json=None, weather_label='weather' ):
    if not os.path.exists(file):
        return set(), set()
    coco_api = COCO(file)
    IDS = list(coco_api.imgs.keys())
    images = set()
    images_ids = set()
    categories = defaultdict(list)
    for e in IDS:
        img = coco_api.loadImgs(e)[0]
        file_name = img['file_name']
        images.add( file_name )
        images_ids.add(img['id'])
        if categories_json is None :
            weather = img[weather_label]
        else:
            weather = categories_json[file_name][weather_label]
            scene = categories_json[file_name]['scene']
            timeofday = categories_json[file_name]['timeofday']
        categories[weather].append(img['id'] )

    return images, images_ids, categories


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
        "info": coco_api.dataset['info'] if 'info' in coco_api.dataset else None,
        'licenses': coco_api.dataset['licenses'] if 'licenses' in coco_api.dataset else None

    }
    # Dump the filtered dictionary to a JSON file
    with open(dest, 'w') as json_file:
        json.dump(filtered_coco_dict, json_file)

def dump_new_foggy_csv(sampled_ids, file, dest, root=None):
    if os.path.exists(dest):
        pressed_key = input("Enter 'c' to continue ")
        if pressed_key != "c":quit()
    coco_api = COCO(file)
    selected_images = [img for img in coco_api.dataset['images'] if img['id'] in sampled_ids]

    for e in selected_images:
        foggy_set = random.choice([0.005, 0.01, 0.02])
        e['file_name'] = e['file_name'] + f"_leftImg8bit_foggy_beta_{foggy_set}" + '.png'
        assert os.path.exists( os.path.join(root, 'train', e['file_name']) )

    selected_annotations = [ann for ann in coco_api.dataset['annotations'] if ann['image_id'] in sampled_ids]
    # Create a new dataset dictionary
    filtered_coco_dict = {
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': coco_api.dataset['categories'],
        "info": coco_api.dataset['info'] if 'info' in coco_api.dataset else None,
        'licenses': coco_api.dataset['licenses'] if 'licenses' in coco_api.dataset else None

    }
    # Dump the filtered dictionary to a JSON file
    with open(dest, 'w') as json_file:
        json.dump(filtered_coco_dict, json_file)

#### Validation of dumped csvs 
ONLY_VAL = False   
DUMP_TRAIN = True   
VAL_DUMP = False 
DUMP_DEBUG = False  
Train_set = 100 
Test_set = 100
DEBUG_COUNT = 20 

####################################################################################################################################################################################
##### DAWN 
DAWN_SUBSET = False 
if DAWN_SUBSET:
    dest_train  = "DATASET/DAWN_train.json"
    dest_test  = "DATASET/DAWN_test.json"
    dest_debug  = "DATASET/DAWN_debug.json"
    dawn_ann = '/data/priyank/synthetic/DAWN2/labels_coco/dawn_labels.json'
    assert os.path.exists(dawn_ann)


####################################################################################################################################################################################
##### WEDGE 
WEDGE_SUBSET = False    
if WEDGE_SUBSET:
    SUBSET_ID = 3
    # dest_train  = f"DATASET/WEDGE_train.json"
    # dest_test  = f"DATASET/WEDGE_test.json"
    dest_train  = f"DATASET/WEDGE_train-{SUBSET_ID}.json"
    dest_test  = f"DATASET/WEDGE_test-{SUBSET_ID}.json"
    dest_debug  = "DATASET/WEDGE_debug.json"
    wedge_ann = '/data/priyank/synthetic/WEDGE/labels_coco/wedge_labels.json'
    Train_set = int(0.8 * 3360)
    print("Train Images : " , Train_set)
    print("Test  Images : " , 3360 - Train_set)
    assert os.path.exists(wedge_ann)

####################################################################################################################################################################################
##### BDDK 
BDD_SUBSET = False        
if BDD_SUBSET:
    SUBSET_ID = 2
    DATASET='/data/priyank/synthetic/bdd100k/labels_coco/'
    dest_train  = f"DATASET/BDDK_train-{SUBSET_ID}.json"
    bdd_ann = f'{DATASET}/bdd100k_labels_images_train_coco.json'

    print("Train Images : " , Train_set)
    assert os.path.exists(bdd_ann)

    dest_debug  = "DATASET/bdd100k_debug-2.json"
    

####################################################################################################################################################################################
##### Foggy Citiscape [Amodal]
Foggy_SUBSET = False 
if Foggy_SUBSET:
    SUBSET_ID = 2 
    folder = 'train'
    Citiscape_ann = '/data/priyank/synthetic/leftImg8bit_foggyDBF/'

    box_type = 'amodal'
    dest_train1  = f"DATASET/Foggy_train-{box_type}-{SUBSET_ID}.json"
    Citiscape_ann1 = os.path.join(Citiscape_ann, 'labels_coco', f'Cityscape_labels_{folder}_{box_type}.json')

    box_type = 'modal'
    dest_train2  = f"DATASET/Foggy_train-{box_type}-{SUBSET_ID}.json"
    Citiscape_ann2 = os.path.join(Citiscape_ann, 'labels_coco', f'Cityscape_labels_{folder}_{box_type}.json')


    print("Train Images : " , Train_set)
    assert os.path.exists(Citiscape_ann1)
    assert os.path.exists(Citiscape_ann2)



####################################################################################################################################################################################
##### BDDK  Category Wise (acc weathers)
BDD_CATEGORY_SUBSET = False         
if BDD_CATEGORY_SUBSET:
    SUBSET_ID = 1
    DATASET='/data/priyank/synthetic/bdd100k/labels_coco/'
    dest_train  = f"DATASET/BDDK_CATEGORY_train-{SUBSET_ID}.json"
    dest_test  = f"DATASET/BDDK_CATEGORY_test-{SUBSET_ID}.json"
    bdd_ann = f'{DATASET}/bdd100k_labels_images_train_coco.json'
    bdd_ann_val = f'{DATASET}/bdd100k_labels_images_val_coco.json'

    print("Train Images : " , Train_set)
    assert os.path.exists(bdd_ann)
    if ONLY_VAL:
        assert False, "Not yet verified"

    if VAL_DUMP:
        weather_json = os.path.join('DATASET/bdd100k_labels_images_val_ANNO.json')
        with open(weather_json) as f:
            train_weather_labels = json.load(f)
        VAL_ANN = bdd_ann_val
    else:
        weather_json = os.path.join('DATASET/bdd100k_labels_images_train_ANNO.json')
        with open(weather_json) as f:
            train_weather_labels = json.load(f)

    weather_label='weather'
    categories_json = train_weather_labels 
    

####################################################################################################################################################################################
##### DAWN Category Wise (acc weathers)
DAWN_CATEGORY_SUBSET = False   
if DAWN_CATEGORY_SUBSET:
    dest_train  = "DATASET/DAWN_CATEGORY_train.json"
    dawn_ann = '/data/priyank/synthetic/DAWN2/labels_coco/dawn_labels.json'
    assert os.path.exists(dawn_ann)


####################################################################################################################################################################################
##### Virtual Kitty 
KITTI_SUBSET = False    
if KITTI_SUBSET:
    SUBSET_ID = 1
    DATASET='/data/priyank/synthetic/VirtualKitti/labels_coco/'

    kitti_ann = f'{DATASET}/kitti_labels_images_train_coco.json'
    kitti_val_ann = f'{DATASET}/kitti_labels_images_val_coco.json'
    if VAL_DUMP and KITTI_SUBSET:
        VAL_ANN = kitti_val_ann
    if DUMP_DEBUG:
        kitti_ann = kitti_val_ann

    dest_train  = f"DATASET/kitti_train-{SUBSET_ID}.json"
    dest_debug  = "DATASET/kitti_debug.json"
    dest_debug  = "DATASET/kitti_debug-2.json"
    dest_test = f"DATASET/kitti_val-{SUBSET_ID}.json"
    
    weather_label='condition'
    categories_json = None 

    assert os.path.exists(kitti_ann)



####################################################################################################################################################################################
##### WIDER FACE 
WIDER_SUBSET = False 
if WIDER_SUBSET:
    SUBSET_ID = 1
    DATASET='/data/priyank/synthetic/WIDER_FACE/labels_coco/'
    wider_face_val_ann = f'{DATASET}/wider_face_labels_images_val_coco.json'
    
    if VAL_DUMP and WIDER_SUBSET:
        VAL_ANN = kitti_val_ann
    if DUMP_DEBUG:
        wider_face_ann = wider_face_val_ann

    
    dest_debug  = "DATASET/WIDER_FACE_debug.json"
    weather_label='condition'
    categories_json = None 
    assert os.path.exists(wider_face_ann)
    # wider_face_labels_images_debug_coco.json

####################################################################################################################################################################################
##### VIS DRONE 2019
VIS_DRONE2019 = False   
if VIS_DRONE2019:
    SUBSET_ID = 1
    DATASET='/data/priyank/synthetic/VisDrone2019-DET-val/labels_coco/'
    vis_drone_val_ann = f'{DATASET}/visiondrone_val_label.json'
    
    if VAL_DUMP and VIS_DRONE2019:
        VAL_ANN = vis_drone_val_ann
    
    dest_debug  = "DATASET/VIS_DRONE2019_debug.json"
    categories_json = None 
    assert os.path.exists(vis_drone_val_ann)
    sampled_ids_train = []
    DEBUG_COUNT = 50

####################################################################################################################################################################################
#####  UAV-benchmark-M 
UAVDT = True   
if UAVDT:
    DATASET='/data/priyank/synthetic/UAVDT/labels_coco/'
    uavdt_train_ann = f'{DATASET}/UAVDT_train_label.json'
    dest_train = f'{DATASET}/UAVDT_train_label-2.json'
    if VAL_DUMP and UAVDT:
        VAL_ANN = uavdt_train_ann
    
    dest_debug  = "DATASET/UAVDT_debug.json"
    categories_json = None 
    assert os.path.exists(uavdt_train_ann)
    sampled_ids_train = []
    Train_set = 40409 // 2
    


if ONLY_VAL:
    ann_path = "DATASET/DAWN_train.json"
    all_images, all_ids = _laod_images(ann_path)
    print(f"Dawn ...  {ann_path} ::: {len(all_ids)}")

    ann_path = f"DATASET/WEDGE_train.json"
    all_images, all_ids = _laod_images(ann_path)
    print(f"WEDGE ... {ann_path} ::: {len(all_ids)}")

    SUBSET_ID = 1
    ann_path = f"DATASET/BDDK_train-{SUBSET_ID}.json"
    all_images, all_ids = _laod_images(ann_path)
    print(f"BDDK ...  {ann_path} ::: {len(all_ids)}")

    SUBSET_ID = 1
    for box_type in ['amodal', 'modal']:
        ann_path  = f"DATASET/Foggy_train-{box_type}-{SUBSET_ID}.json"
        all_images, all_ids = _laod_images(ann_path)
        print(f"Foggy Citiscape [{box_type}]... {ann_path} ::: {len(all_ids)}")
    
    # Dawn ...  DATASET/DAWN_train.json ::: 100
    # WEDGE ... DATASET/WEDGE_train.json ::: 2688
    # BDDK ...  DATASET/BDDK_train-1.json ::: 100
    # Foggy Citiscape [amodal]... DATASET/Foggy_train-amodal-1.json ::: 100
    # Foggy Citiscape [modal]... DATASET/Foggy_train-modal-1.json ::: 100

    
    
    quit()
    
if DAWN_SUBSET:
    all_images, all_ids = _laod_images(dawn_ann)
    ann = dawn_ann
elif WEDGE_SUBSET:
    all_images, all_ids = _laod_images(wedge_ann)
    ann = wedge_ann
elif BDD_SUBSET:
    all_images, all_ids = _laod_images(bdd_ann)
    ann = bdd_ann
elif Foggy_SUBSET:
    all_images, all_ids = _laod_images(Citiscape_ann1)
    ann = Citiscape_ann1
elif BDD_CATEGORY_SUBSET and (DUMP_TRAIN or DUMP_DEBUG):
    all_images, all_ids, categories = _laod_images_with_categories(bdd_ann, categories_json=train_weather_labels)
    ann = bdd_ann
elif DAWN_CATEGORY_SUBSET:
    all_images, all_ids , categories= _laod_images_with_categories(dawn_ann)
    ann = dawn_ann
elif KITTI_SUBSET:
    all_images, all_ids = _laod_images(kitti_ann)
    ann = kitti_ann
elif WIDER_SUBSET:
    all_images, all_ids = _laod_images(wider_face_ann)
    ann = wider_face_ann
elif VIS_DRONE2019:
    all_images, all_ids = _laod_images(vis_drone_val_ann)
    ann = vis_drone_val_ann
elif UAVDT:
    all_images, all_ids = _laod_images(uavdt_train_ann)
    ann = uavdt_train_ann

print("****", ann)

###### Train 
if DUMP_TRAIN:
    if BDD_CATEGORY_SUBSET or DAWN_CATEGORY_SUBSET:
        sampled_ids_train = [ ]
        for categ in categories:
            category_ids = random.sample(categories[categ], Train_set)
            sampled_ids_train += category_ids
            assert len(set(category_ids)) == Train_set
        dump_new_csv(sampled_ids_train, ann, dest_train)
    else:
        sampled_ids_train = random.sample(all_ids, Train_set)
        assert len(set(sampled_ids_train)) == Train_set
        if Foggy_SUBSET:
            dump_new_foggy_csv(sampled_ids_train, Citiscape_ann1, dest_train1, root=Citiscape_ann)
            dump_new_foggy_csv(sampled_ids_train, Citiscape_ann2, dest_train2, root=Citiscape_ann)
        else:
            dump_new_csv(sampled_ids_train, ann, dest_train)

###### Test / Val 
if VAL_DUMP:
    if KITTI_SUBSET or BDD_CATEGORY_SUBSET:
        all_val_images, all_val_ids , categories_val = _laod_images_with_categories(VAL_ANN, categories_json=categories_json, weather_label=weather_label)
        sampled_ids_test = []
        for categ in categories_val:
            count = min(Test_set, len(categories_val[categ]))
            category_ids = random.sample(categories_val[categ], count )
            assert len(set(category_ids)) == count
            sampled_ids_test += category_ids
        dump_new_csv(sampled_ids_test, VAL_ANN, dest_test)
    else:
        sampled_ids_test = [id for id in all_ids if id not in sampled_ids_train]
        assert set(sampled_ids_test).intersection(set(sampled_ids_train)) == set()
        assert len(set(sampled_ids_test)) +  len(set(sampled_ids_train))   == len(all_ids)
        dump_new_csv(sampled_ids_test, ann, dest_test)


###### Debug 
if DUMP_DEBUG:
    # sampled_ids_debug = random.sample(all_ids, 100)
    # assert len(set(sampled_ids_debug)) == 100
    sampled_ids_debug = random.sample(all_ids, DEBUG_COUNT)
    dump_new_csv(sampled_ids_debug, ann, dest_debug)




# conda activate GLIP 
# cd ~/robustness_object_detection/
# python Scripts/external_subset_generator.py

# cp DATASET/DAWN_*.json /data/priyank/synthetic/DAWN2/labels_coco/
# cp DATASET/WEDGE_*.json /data/priyank/synthetic/WEDGE/labels_coco/


# cp DATASET/WEDGE_*.json /home/c3-0/datasets/WEDGE/labels_coco/


