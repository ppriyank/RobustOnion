# DATASET 

Flicker and COCO, LVIS and ODWIN needs to be downloaded from office website. 
Flicker30K is used for training and COCO for all evalaution.
Flicker and ODINW-35 folder names have been changed. Apologies.

Dataset annotations are present in `DATASET/`. Flicker subset generated using [check_flickr.py](check_flickr.py)


for Debugging 
```
ln -s "original folder path" "shortcut name"
ln -s ~/anaconda3/envs/OD/lib/python3.9/site-packages/pycocotools/ ~/robustness_object_detection/MMGDINO/

export DETECTRON2_DATASETS=/data/priyank/synthetic/flickr_dataset_30k/
export DATASET=/data/priyank/synthetic/flickr_dataset_30k/
### path of COCO ROOT Folder / LVIS / ODINW-13
export ADD_DATASET=/data/priyank/synthetic/ 
```

#### BDD100K 
Download from `Kaggle` (https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)

```
cd ~/robustness_object_detection/
python Scripts/bdd2coco.py --bdd_dir /data/priyank/synthetic/bdd100k/
```

#### DAWN 
- Dataset [Link](https://data.mendeley.com/datasets/766ygrbt8y/3) 
 
- Convert to Coco annotation [Link](Scripts/DAWN2coco.py)


#### WEDGE 

- Dataset [Link](https://github.com/Infernolia/WEDGE/tree/main/Dataset/WEDGE) [Gdrive](https://drive.google.com/file/d/1gmnoZWw9Oh-A60HE_qhXlwsNj2n1-iZ3/view)
 
- Convert to Coco annotation [Link](Scripts/WEDGE2coco.py)


#### FoggyCityScape 

- Dataset [Link](https://www.cityscapes-dataset.com/downloads/)
   - leftImg8bit_trainval_foggyDBF.zip (20GB) [md5]
   - gtBbox_cityPersons_trainval.zip (2.2MB) [md5]

- Convert to Coco annotation [Link](Scripts/FoggyCityscape2coco.py)

#### Adver-City

- Dataset [Link](https://github.com/QUARRG/Adver-City?tab=readme-ov-file) [Line2](https://labs.cs.queensu.ca/quarrg/datasets/adver-city/)

You can simulate the dataset. We have not used this dataset 


### Adding new Dataset 

- Step 1: Create Coco format annotation (e.g. `DAWN2coco.py`,  `bdd2coco.py`,  `FoggyCityscape2coco.py`, `virtual_kitt2coco.py`, `WEDGE2coco.py`)
- Step 2: Add dataset paths in  `GLIP/maskrcnn_benchmark/config/paths_catalog.py`
- Step 3: Create Dataset Class in  `GLIP/maskrcnn_benchmark/data/datasets/external_dataset.py`
- Step 4: Register the dataset class in  `GLIP/maskrcnn_benchmark/data/datasets/__init__.py`
- Step 5: Add dataset class to redirect evaluation to custom per category eval `GLIP/maskrcnn_benchmark/data/datasets/evaluation/__init__.py`
- Step 6: Patch Evaluation function to evaluate per category ('weather' by default) `GLIP/maskrcnn_benchmark/data/datasets/evaluation/dawn/dawn_eval.py`
- Step 7: Update external dataset in `GLIP/maskrcnn_benchmark/engine/inference.py`
- Training requires "grounding dataset"



