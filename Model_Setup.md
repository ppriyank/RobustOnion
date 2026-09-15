
# Models 

### GLIP 

[GITHUB](https://github.com/microsoft/GLIP.git)
```
cd ~/robustness_object_detection/
conda create --name OD python=3.9 -y
conda activate OD 
<!-- module load cuda/12.5  -->
python -m pip install numpy==1.26.4
python -m pip install albumentations==2.0.5


python -c "import torch; print(torch.__version__);"
python -m pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install packaging
python -c "import torch; print(torch.__version__);" #### 1.10


python -c "import setuptools"
python -m pip install setuptools==59.8.0
python -m pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
python -m pip install transformers diffdist 
python -m pip install git+https://github.com/lvis-dataset/lvis-api.git
python -m pip install vlkit inflect nltk umap-learn keyboard
python -m pip install -U scikit-image keyboard ultralytics pycocotools scikit-learn noise
python -m pip install salesforce-lavis 
python -m pip install timm==1.0.15
python -m pip install openmim
python -m pip install numpy==1.26.4
python -m pip install transformers==4.39.3 
python -m pip install wordcloud seaborn

cd ~/robustness_object_detection/GLIP
CC=gcc-9 CXX=g++-9 python setup.py clean --all build develop --user
```


For `H100 sm_90` || `torch > 1.10` (fix uploaded here : https://discuss.pytorch.org/t/maskrcnn-benchmark-sm-90-cuda-11-8/218350/2)

```
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 albumentations==2.0.5 setuptools==59.8.0
python -m pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers diffdist 
python -m pip install seaborn vlkit pandas
python -m pip install -U scikit-image
python -m pip install ultralytics pycocotools scikit-learn noise

cd ~/robustness_object_detection/GLIP
python setup.py clean --all build develop --user
```


Model Weights 
```
GLIP/MODEL/glip_a_tiny_o365.pth
GLIP/MODEL/glip_large_model.pth
GLIP/MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
GLIP/MODEL/glip_tiny_model_o365_goldg.pth
GLIP/MODEL/glip_tiny_model_o365.pth
GLIP/MODEL/jx_vit_base_patch16_224_in21k-e5005f0a.pth
GLIP/MODEL/swin_tiny_patch4_window7_224.pth
```

### MGDINO

[GITHUB](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)

Taken from https://github.com/open-mmlab/mmdetection/tree/main

```
conda activate OD 
python -m pip install openmim

~/mambaforge/envs/OD/bin/mim install mmengine 
~/mambaforge/envs/OD/bin/mim install "mmcv==2.0.0" 
~/mambaforge/envs/OD/bin/mim install mmdet

mim install mmengine 
mim install "mmcv==2.0.0"
<!-- mim install "mmcv==2.1.0" -->
mim install mmdet

python -m pip install fairscale termcolor
python -m pip install transformers==4.39.3 
```

Download weights 

```
MMGDINO/WTS/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
MMGDINO/WTS/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
MMGDINO/WTS/grounding_dino_swin-l_pretrain_all-56d69e78.pth
MMGDINO/WTS/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
MMGDINO/WTS/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth
MMGDINO/WTS/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth
MMGDINO/WTS/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
MMGDINO/WTS/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth
MMGDINO/WTS/groundingdino_swint_ogc_mmdet-822d7e9d.pth
```


### GLEE
Taken from https://github.com/FoundationVision/GLEE.git

```
cd ~/robustness_object_detection/GLEE/

conda create --name GLEE python=3.9 -y
conda activate GLEE 
<!-- module load cuda/12.1 -->

python -m pip install shapely==1.7.1 lvis scipy fairscale einops xformers
python -m pip install opencv-python-headless tensorboard timm ftfy transformers==4.36.0 
<!-- export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}} -->
<!-- export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} -->
python -m pip install setuptools==57.5.0
<!-- python -m pip install setuptools==75.8.0 -->
python -m pip install git+https://github.com/facebookresearch/detectron2.git
python -m pip install numpy==1.26.4 gradio pillow==9.5.0

python -m pip install xformers
python -m pip install git+https://github.com/facebookresearch/fvcore.git

python -m pip install albumentations==2.0.5 scikit-learn noise
python -m pip install -U scikit-image
python -m pip install pycocotools

<!-- https://github.com/Epiphqny/VisTR/issues/5#issuecomment-819837393 -->
ls ~/anaconda3/envs/GLEE/lib/python3.9/site-packages/pycocotools/
cp pycocotools/* ~/anaconda3/envs/GLEE/lib/python3.9/site-packages/pycocotools/
cp pycocotools/* ~/mambaforge/envs/GLEE/lib/python3.9/site-packages/pycocotools/
cp pycocotools/* ~/.local/lib/python3.9/site-packages/pycocotools-2.0.8-py3.9-linux-x86_64.egg/pycocotools/
cp pycocotools/* ~/.conda/envs/GLEE/lib/python3.9/site-packages/pycocotools/


wget  -P projects/GLEE/clip_vit_base_patch32/  https://huggingface.co/spaces/Junfeng5/GLEE_demo/resolve/main/GLEE/clip_vit_base_patch32/pytorch_model.bin   

wget  -P projects/GLEE/clip_vit_base_patch32/ https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt

wget  -P Weights/ https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
```


Download weights in GLEE/Weights

```
GLEE/Weights/converted_EVA02_m38m_psz14to16.pth
GLEE/Weights/eva02_L_pt_m38m_p14to16.pt
GLEE/Weights/GLEE_Lite_joint.pth
GLEE/Weights/GLEE_Lite_pretrain.pth
GLEE/Weights/GLEE_Lite_scaleup.pth
GLEE/Weights/GLEE_Plus_joint.pth
GLEE/Weights/GLEE_Plus_pretrain.pth
GLEE/Weights/GLEE_Plus_scaleup.pth
GLEE/Weights/GLEE_Pro_joint.pth
GLEE/Weights/GLEE_Pro_pretrain.pth
GLEE/Weights/GLEE_Pro_scaleup.pth
GLEE/Weights/swin_large_patch4_window12_384_22k.pth
GLEE/Weights/swin_tiny_patch4_window7_224.pth
GLEE/projects/GLEE/clip_vit_base_patch32/pytorch_model.bin
```





