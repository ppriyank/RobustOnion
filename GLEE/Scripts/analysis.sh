
cd ~/robustness_object_detection/GLEE/
conda activate GLEE2
export DETECTRON2_DATASETS=/home/c3-0/datasets/
export DETECTRON2_DATASETS=/data/priyank/synthetic/
NUM_GPU=1
GPU_DEVICE=1

################################### LITE 
## GLLE_LITE_JOINT (STAGE 1) 
CONFIG=projects/GLEE/configs/images/Lite/Stage1_pretrain_openimage_obj365_CLIPfrozen_R50.yaml 
WT=Weights/GLEE_Lite_pretrain.pth
OUTPUT=GLLE_LITE_STAGE1

# GLLE_LITE_JOINT (STAGE 2) 
CONFIG=projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml
WT=Weights/GLEE_Lite_joint.pth
OUTPUT=GLLE_LITE_STAGE2

# GLLE_LITE_JOINT (STAGE 3) 
CONFIG=projects/GLEE/configs/images/Lite/Stage3_scaleup_CLIPteacher_R50.yaml 
WT=Weights/GLEE_Lite_scaleup.pth
OUTPUT=GLLE_LITE_STAGE3

#################################### PLUS 
MODEL_NAME=D2SwinTransformer_Dump 
# GLLE_PLUS_JOINT (STAGE 1) 
CONFIG=projects/GLEE/configs/images/Plus/Stage1_pretrain_openimage_obj365_CLIPfrozen_SwinL.yaml 
WT=Weights/GLEE_Plus_pretrain.pth
OUTPUT=GLLE_PLUS_STAGE1

# GLLE_PLUS_JOINT (STAGE 2) 
CONFIG=projects/GLEE/configs/images/Plus/Stage2_joint_training_CLIPteacher_SwinL.yaml
WT=Weights/GLEE_Plus_joint.pth
OUTPUT=GLLE_PLUS_STAGE2

# GLLE_PLUS_JOINT (STAGE 3) 
CONFIG=projects/GLEE/configs/images/Plus/Stage3_scaleup_CLIPteacher_SwinL.yaml
WT=Weights/GLEE_Plus_scaleup.pth
OUTPUT=GLLE_PLUS_STAGE3

#################################### PRO 
MODEL_NAME=D2_EVA02_DUMP 

# GLLE_PRO_JOINT (STAGE 1) 
CONFIG=projects/GLEE/configs/images/Pro/Stage1_pretrain_openimage_obj365_CLIPfrozen_EVA02L_LSJ1536.yaml 
WT=Weights/GLEE_Pro_pretrain.pth
OUTPUT=GLLE_PRO_STAGE1

# GLLE_PRO_JOINT (STAGE 2) 
CONFIG=projects/GLEE/configs/images/Pro/Stage2_joint_training_CLIPteacher_EVA02L.yaml
WT=Weights/GLEE_Pro_joint.pth
OUTPUT=GLLE_PRO_STAGE2

# GLLE_PRO_JOINT (STAGE 3) 
CONFIG=projects/GLEE/configs/images/Pro/Stage3_scaleup_CLIPteacher_EVA02L.yaml
WT=Weights/GLEE_Pro_scaleup.pth
OUTPUT=GLLE_PRO_STAGE3


########################################################################
###### COCO DUMP FETURES # Spatial Tokens (Dumps only 0th layer)
########################################################################
DATALOADER=RefCOCODatasetMapper_Perturb

SEV=3
NOISES=("low-res" "motion_blur2")
for MODE in "${NOISES[@]}"
do
  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats \
        OUTPUT_DIR $OUTPUT/$MODE MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV \
        MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME $MODEL_NAME \
        TEST.DUMP_INDEX 0 TEST.DUMP_MODE 'backbone_only' 
done
# --dump-start 0 --dump-end 10 

MODE="atmospheric"
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
        OUTPUT_DIR $OUTPUT/$MODE MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV \
        MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME $MODEL_NAME \
        TEST.DUMP_INDEX 0 TEST.DUMP_MODE 'backbone_only' 

MODE=NONE
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
        OUTPUT_DIR $OUTPUT/$MODE MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER \
        MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME $MODEL_NAME \
        TEST.DUMP_INDEX 0 TEST.DUMP_MODE 'backbone_only' 

