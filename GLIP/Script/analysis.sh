# srun --pty --gres=gpu:2 --cpus-per-gpu=8 -C gmem48 bash 
conda activate OD

cd ~/robustness_object_detection/GLIP/
ENV='nccl'
NUM_GPU=1
PORT=12345
mkdir ucf_output

export DETECTRON2_DATASETS=/home/c3-0/datasets/flickr_dataset_30k/
export DATASET=/home/c3-0/datasets/flickr_dataset_30k/
export ADD_DATASET=/home/c3-0/datasets/
export TOKENIZERS_PARALLELISM=true 


export DETECTRON2_DATASETS=/data/priyank/synthetic/flickr_dataset_30k/
export DATASET=/data/priyank/synthetic/flickr_dataset_30k/
export ADD_DATASET=/data/priyank/synthetic/
export TOKENIZERS_PARALLELISM=true 



CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output/GLIP-T_5.txt


##############################################################################################################
###################### DUMP everything  (COCO)  ######################
DATALOADER=COCODataset_Perturb
TECHNIQUE=GeneralizedVLRCNN_DUMPING


MODE="low-res"
MODE="motion_blur2"

DUMP=GLIP-T_5-NECK_SEP-$MODE
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU OUTPUT_DIR $DUMP MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.RPN.RETURN_FUSED_FEATURES True \
  MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "ALL_DUMP" TEST.DUMP_INDEX 0,2000 

CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU OUTPUT_DIR $DUMP MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.RPN.RETURN_FUSED_FEATURES True \
  MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "ALL_DUMP" TEST.DUMP_INDEX 2000,4000

CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU OUTPUT_DIR $DUMP MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.RPN.RETURN_FUSED_FEATURES True \
  MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "ALL_DUMP" TEST.DUMP_INDEX 4000,6000 


############ Atmospheric 
MODE="atmospheric"
DUMP=GLIP-T_5-NECK_SEP-$MODE
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
  TEST.IMS_PER_BATCH $NUM_GPU OUTPUT_DIR $DUMP MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.RPN.RETURN_FUSED_FEATURES True \
  MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "ALL_DUMP" TEST.DUMP_INDEX 0,2000 

CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
  TEST.IMS_PER_BATCH $NUM_GPU OUTPUT_DIR $DUMP MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.RPN.RETURN_FUSED_FEATURES True \
  MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "ALL_DUMP" TEST.DUMP_INDEX 2000,4000

CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
  TEST.IMS_PER_BATCH $NUM_GPU OUTPUT_DIR $DUMP MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.RPN.RETURN_FUSED_FEATURES True \
  MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "ALL_DUMP" TEST.DUMP_INDEX 4000,6000 

  # DATASETS.TEST '("bdd100k_val",)' 




##############################################################################################################
###################### Visualize Prediction ######################
PORT=12346
DATALOADER=COCODataset_Perturb

MODE="NONE"
DUMP=GLIP-T_5-PRED-$MODE
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $DUMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "PRED_DUMP" 


MODE="low-res"
MODE="motion_blur2"
DUMP=GLIP-T_5-PRED-$MODE
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $DUMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "PRED_DUMP" 
  

MODE="atmospheric"
DUMP=GLIP-T_5-PRED-$MODE
CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $DUMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  TEST.PREDICT "PRED_DUMP" 






##############################################################################################################
####### FLOP AND PARAM COUNT  --GFLOP

###### VANILLA MODEL 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  --GFLOP TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP 
# ***** FLOP TOTAL : 34.722541056
#  Model parameters: 231,780,222
#  Backbone parameters: 27,520,506

###### LR-TK0 (vanilla 32 x 32 )
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
BACKBONE_TYPE=LR_TKO
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  --GFLOP TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE 
# ***** FLOP TOTAL : 34.730713536
#  Model parameters: 233,353,086
#  Backbone parameters: 29,093,370

###### LR-TK0 (Original LrTK0 600/4 x 600/4 , 600/8 x 600/8, 600/16 x 600/16, 600/32 x 600/32 )
# MODEL.BACKBONE.PROMPT_NUM_TOKEN
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
BACKBONE_TYPE=LR_TKO
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  --GFLOP TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE \
  MODEL.BACKBONE.PROMPT_NUM_TOKEN 150,75,37,19
# ***** FLOP TOTAL : 34.730713536
#  Model parameters: 237,983,166
#  Backbone parameters: 33,723,450






##############################################################################################################
###################### Dump Mean Zero-shot Feats ########################################################################################
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output/GLIP-T_5.txt
###### local path with 100 samples 
DATASET=/home/priyank/robustness_object_detection/DATASET/

TECHNIQUE=GeneralizedVLRCNN_TRILANG_DUMP 
DATALOADER=COCODataset_Perturb
NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
for MODE in "${NOISES[@]}"
do
  ###### All feats
  # CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  #   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
  #   MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "Feat_DUMP"
  
  ###### Spatial Tokens (Dumps only 0th layer)
  CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP"
done

NOISES=("snow_model2" "atmospheric" "rain_model2")
for MODE in "${NOISES[@]}"
do
  CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "Feat_DUMP"
done

###### Spatial Tokens No Noise 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE NONE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP"


###### BBDK100 + Dump 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
  MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP_CATEGORY" DATASETS.TEST '("bdd100k_category_train",)'
  
# DATASETS.TEST '("bdd100k_train",)'  '("bdd100k_train2",)'  
# TEST.PREDICT "Feat_DUMP"  "SPATIAL_DUMP_CATEGORY" "SPATIAL_DUMP"


###### DAWN + Dump 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
  MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP_CATEGORY" DATASETS.TEST '("dawn_category_train",)' 

# dawn_category_train
# DATASETS.TEST '("dawn_train",)' 
# TEST.PREDICT "Feat_DUMP"  "SPATIAL_DUMP_CATEGORY" "SPATIAL_DUMP"


###### Foggy + Amodal + Dump 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
  MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "Feat_DUMP" DATASETS.TEST '("FoggyCityscape_amodal_train",)' 


###### Foggy + modal + Dump 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
  MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "Feat_DUMP" DATASETS.TEST '("FoggyCityscape_modal_train",)' 






##############################################################################################################
###################### Noise Trained Model Features Features ########################################################################################
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
MODEL_TYPE=GLIP-T_5
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
TECHNIQUE=GeneralizedVLRCNN_TRILANG_DUMP 
DATALOADER=COCODataset_Perturb
DUMP_DEST=~/robustness_object_detection/Analysis/Feats/Noise-Tra_Spa_Feat
BACKBONE_TYPE=LR_TKO

NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
for MODE in "${NOISES[@]}"
do
  DUMP_FOLDER=EXPS/$MODEL_TYPE-$BACKBONE_TYPE
  DUMP_FOLDER=$DUMP_FOLDER+$MODE
  WT=$DUMP_FOLDER/model_best.pth
  ls $WT
  for LOCAL_MODE in "${NOISES[@]}"
    do  
    ###### Spatial Tokens (Dumps only 0th layer) & Features interpolated 
    CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
      TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $DUMP_DEST  \
      DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $LOCAL_MODE MODEL.META_ARCHITECTURE $TECHNIQUE \
      TEST.PREDICT "SPATIAL_DUMP_RESIZE" MODEL.WEIGHT $WT
    done 
done


LOCAL_NOISES=("snow_model2" "atmospheric" "rain_model2")
NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
for MODE in "${NOISES[@]}"
do
  DUMP_FOLDER=EXPS/$MODEL_TYPE-$BACKBONE_TYPE
  DUMP_FOLDER=$DUMP_FOLDER+$MODE
  WT=$DUMP_FOLDER/model_best.pth
  ls $WT
  for LOCAL_MODE in "${LOCAL_NOISES[@]}"
    do  
    ###### Spatial Tokens (Dumps only 0th layer) & Features interpolated 
    CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
      TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $DUMP_DEST  \
      DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $LOCAL_MODE MODEL.META_ARCHITECTURE $TECHNIQUE \
      TEST.PREDICT "SPATIAL_DUMP_RESIZE" MODEL.WEIGHT $WT
    done 
done

# LOCAL_NOISES=( "" "")
MODE="rain_model2"
DUMP_FOLDER=EXPS/$MODEL_TYPE-$BACKBONE_TYPE
DUMP_FOLDER=$DUMP_FOLDER+$MODE
WT=$DUMP_FOLDER/model_best.pth
ls $WT
LOCAL_MODE="motion_blur2"

###### Spatial Tokens (Dumps only 0th layer) & Features interpolated 
CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP  \
  TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $DUMP_DEST  \
  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $LOCAL_MODE MODEL.META_ARCHITECTURE $TECHNIQUE \
  TEST.PREDICT "SPATIAL_DUMP_RESIZE" MODEL.WEIGHT $WT
    
    

NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog" "snow_model2" "atmospheric" "rain_model2")
for MODE in "${NOISES[@]}"
do
  for LOCAL_MODE in "${NOISES[@]}"
  do
    Feats=~/robustness_object_detection/Analysis/Feats/Noise-Tra_Spa_Feat
    Feats=$Feats/GLIP-T_5-LR_TKO+$MODE"_"SPATIAL_coco_$LOCAL_MODE.pth
    ls $Feats
    done 
done









##############################################################################################################
###################### All models backbone [0] feats (no mean) ########################################################################################
TECHNIQUE=GeneralizedVLRCNN_TRILANG_DUMP 
DATALOADER=COCODataset_Perturb
GPU_DEVICE=1

CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT_DIR=T_5

CONFIG=configs/pretrain/glip_A_Swin_T_O365.yaml
WT=MODEL/glip_a_tiny_o365.pth
OUTPUT_DIR=T_A

CONFIG=configs/pretrain/glip_Swin_T_O365.yaml
WT=MODEL/glip_tiny_model_o365.pth
OUTPUT_DIR=T_B

CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT=MODEL/glip_tiny_model_o365_goldg.pth
OUTPUT_DIR=T_C

CONFIG=configs/pretrain/glip_Swin_L.yaml
WT=MODEL/glip_large_model.pth
OUTPUT_DIR=T_L




NOISES=("low-res" "motion_blur2")
for MODE in "${NOISES[@]}"
do
  ###### Spatial Tokens (Dumps only 0th layer)
  CUDA_VISIBLE_DEVICES=$GPU_DEVICE PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $OUTPUT_DIR  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP_RESIZE_NO_BICU"
done

NOISES=("atmospheric")
for MODE in "${NOISES[@]}"
do
  CUDA_VISIBLE_DEVICES=$GPU_DEVICE PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $OUTPUT_DIR  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP_RESIZE_NO_BICU"
done

###### Spatial Tokens No Noise 
CUDA_VISIBLE_DEVICES=$GPU_DEVICE PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR $OUTPUT_DIR  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE NONE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "SPATIAL_DUMP_RESIZE_NO_BICU"

