import os
import json
import argparse
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(description='wider2coco')
parser.add_argument('--wider_dir', type=str, default='E:\\bdd100k')
cfg = parser.parse_args()

src_val_dir = os.path.join(cfg.wider_dir, 'wider_face_split', 'wider_face_val_bbx_gt.txt')
src_train_dir = os.path.join(cfg.wider_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt')

os.makedirs(os.path.join(cfg.wider_dir, 'labels_coco'), exist_ok=True)

dst_val_dir = os.path.join(cfg.wider_dir, 'labels_coco', 'wider_face_labels_images_val_coco.json')
dst_train_dir = os.path.join(cfg.wider_dir, 'labels_coco', 'wider_face_labels_images_train_coco.json')



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



_LICENSE = "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)"
_HOMEPAGE = "http://shuoyang1213.me/WIDERFACE/"
_CITATION = """\
@inproceedings{yang2016wider,
    Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
    Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Title = {WIDER FACE: A Face Detection Benchmark},
    Year = {2016}}
"""

def wedge2coco_detection(split, save_dir, annot_fname=None, mode='train',):
    
    attr_dict = {"categories": [{"supercategory": "none", "id": 1, "name": "face"},] }
    id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
    id_counter = {i['name']: 0 for i in attr_dict['categories']}

    blur_class = ["clear", "normal", "heavy"]
    expression = ["typical", "exaggerate"]
    illumination = ["normal", "exaggerate "]
    occlusion = ["no", "partial", "heavy"]
    pose = ["typical", "atypical"]

    images = list()
    annotations = list()
    ignore_categories = set()

    data = _generate_examples(split=split, data_dir=cfg.wider_dir, annot_fname=annot_fname)
    
    counter = 0
    local_counter = 0 
    for i in tqdm(data.keys()):
        counter += 1
        image = dict()
        
        image['file_name'] = data[i]['image'].replace(cfg.wider_dir, "")
        img = Image.open(data[i]['image'])
        
        image['height'] = img.height
        image['width'] = img.width
        image['id'] = counter
        assert i == counter -1 
        empty_image = True
        
        for l in data[i]['faces']:
            if l['invalid'] == 1:
                continue 
            annotation = dict()
            empty_image = False
            annotation["iscrowd"] = 0
            annotation["image_id"] = image['id']
            annotation['bbox'] = l['bbox']
            annotation['area'] = float( l['bbox'][2] * l['bbox'][3] )
            
            annotation['category_id'] = id_dict['face']
            annotation['ignore'] = 0
            annotation['id'] = local_counter
            local_counter += 1
            annotation['segmentation'] = []

            
            annotation['blur'] = blur_class[l['blur']]
            annotation['expression'] = expression[l['expression']]
            annotation['illumination'] = illumination[l['illumination']]
            annotation['occlusion'] = occlusion[l['occlusion']]
            annotation['pose'] = pose[l['pose']]
            
            annotations.append(annotation)
            id_counter[ 'face' ] += 1
            
        
        if empty_image:
            print('empty image!')
            continue

        # display_box_over_img(image['file_name'], annotations, root=cfg.wider_dir)
        # quit()
        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('ignored categories: ', ignore_categories)
    print("Counter of categories", id_counter)
    print('saving...')

    with open(save_dir, "w") as file:
        json.dump(attr_dict, file)
        print('Done.')


# https://huggingface.co/datasets/CUHK-CSE/wider_face/blob/main/wider_face.py
def _generate_examples(split, data_dir, annot_fname):
    data = {}
    image_dir = os.path.join(data_dir, "WIDER_" + split, "images")
    with open(annot_fname, "r", encoding="utf-8") as f:
        idx = 0
        while True:
            line = f.readline()
            line = line.rstrip()
            if not line.endswith(".jpg"):
                break
            image_file_path = os.path.join(image_dir, line)
            assert os.path.exists(image_file_path), "Image doesnt exist"
            faces = []
            if split != "test":
                # Read number of bounding boxes
                nbboxes = int(f.readline())
                # Cases with 0 bounding boxes, still have one line with all zeros.
                # So we have to read it and discard it.
                if nbboxes == 0:
                    f.readline()
                else:
                    for _ in range(nbboxes):
                        line = f.readline()
                        line = line.rstrip()
                        line_split = line.split()
                        assert len(line_split) == 10, f"Cannot parse line: {line_split}"
                        line_parsed = [int(n) for n in line_split]
                        (
                            xmin,
                            ymin,
                            wbox,
                            hbox,
                            blur,
                            expression,
                            illumination,
                            invalid,
                            occlusion,
                            pose,
                        ) = line_parsed
                        faces.append(
                            {
                                "bbox": [xmin, ymin, wbox, hbox],
                                "blur": blur,
                                "expression": expression,
                                "illumination": illumination,
                                "occlusion": occlusion,
                                "pose": pose,
                                "invalid": invalid,
                            }
                        )
            assert idx not in data 
            data[idx] ={"image": image_file_path, "faces": faces}
            idx += 1
    return data 


def main(val_only=False):
    if not val_only:
        print('Loading training set...')
        assert False, "not yet verified...."
        
    print('Loading validation set...')
    with open(src_val_dir) as f:
        wedge2coco_detection(split='val', save_dir=dst_val_dir, annot_fname=src_val_dir, mode='val')

    # ignored categories:  set()
    # Counter of categories {'face': 39123}





if __name__ == '__main__':
  main(val_only=True )


# cd ~/robustness_object_detection/
# python Scripts/wider2coco.py --wider_dir /data/priyank/synthetic/WIDER_FACE/

