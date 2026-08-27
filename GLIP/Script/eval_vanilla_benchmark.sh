#!/bin/bash

#SBATCH --job-name=OR_1
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH --exclude=c4-4
#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G --cpus-per-gpu=8


################SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G -c8
###############SBATCH -p gpu --qos=day
##############SBATCH -p gpu --qos=short 

# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --qos preempt bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --exclude=c4-4 bash 
# srun --pty --gres=gpu:2 --cpus-per-gpu=8 -C gmem48 bash 


nvidia-smi
source ~/.bashrc
CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh ucf_output/
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/robustness_object_detection/GLIP/ucf_output/

module load cuda/11.3
conda activate OD
cd ~/robustness_object_detection/GLIP/


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

export DETECTRON2_DATASETS=/home/c3-0/datasets/flickr_dataset_30k/
export DATASET=/home/c3-0/datasets/flickr_dataset_30k/
export ADD_DATASET=/home/c3-0/datasets/
export TOKENIZERS_PARALLELISM=true 

NUM_GPU=1


if [[ "$SLURM_JOB_NODELIST" == "c4-4" ]]; then
        sbatch ucf_output/slurm-$SLURM_JOB_ID.sh
        exit 1
fi



###################### GLIP-T [5] ########################################################################################
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output/GLIP-T_5.txt

# ###################### GLIP-T A ##################################################################
# CONFIG=configs/pretrain/glip_A_Swin_T_O365.yaml
# WT=MODEL/glip_a_tiny_o365.pth
# OUTPUT=ucf_output/GLIP-A.txt

# ###################### GLIP-T B ##################################################################
# CONFIG=configs/pretrain/glip_Swin_T_O365.yaml
# WT=MODEL/glip_tiny_model_o365.pth
# WT=/home/c3-0/datasets/obj_det_robustness/weights/GLIP/glip_tiny_model_o365.pth
# OUTPUT=ucf_output/GLIP-B.txt

# ###################### GLIP-T C ############################################################
# CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
# WT=MODEL/glip_tiny_model_o365_goldg.pth
# WT=/home/c3-0/datasets/obj_det_robustness/weights/GLIP/glip_tiny_model_o365_goldg.pth
# OUTPUT=ucf_output/GLIP-C.txt

# ###################### GLIP-L [7] ##################################################################
# CONFIG=configs/pretrain/glip_Swin_L.yaml
# WT=MODEL/glip_large_model.pth
# WT=/home/c3-0/datasets/obj_det_robustness/weights/GLIP/glip_large_model.pth
# OUTPUT=ucf_output/GLIP-L.txt




# ###### COCO 
# printf "\n\n #### OG \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  >> $OUTPUT


##### Perturbations
# DATALOADER=COCODataset_Perturb

# ###### Low Res 
# MODE="low-res"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### Focus Blur 
# MODE="focus_blur"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT
 
# ###### Pixel Dropout
# MODE="pixel_dropout"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### Fog 
# MODE="fog"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### Gaussian Blur 
# MODE="gaussian_blur"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### Chromatic  
# MODE="chromatic"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### ISO Blur 
# MODE="iso_blur"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### JPG Compression 
# MODE="jpg_compression"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### Salt-pepper 
# MODE="salt_pepper"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

# ###### Atmospheric 
# MODE="atmospheric"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATALOADER.NUM_WORKERS 0 >> $OUTPUT

# ###### Rain  
# MODE="rain_model2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATALOADER.NUM_WORKERS 0 >> $OUTPUT

# ###### SNOW 
# MODE="snow_model2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATALOADER.NUM_WORKERS 0 >> $OUTPUT


# ###### Motion Blur 
# MODE="motion_blur2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT


# ##### BBDK100
# printf "\n\n #### BBDK \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("bdd100k_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False >> $OUTPUT

  
  

# # ###### DAWN
# printf "\n\n #### DAWN \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("dawn_test",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False  >> $OUTPUT


# # ###### WEDGE
# printf "\n\n #### WEDGE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("wedge_test",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False  >> $OUTPUT




###################### FoggyCitiscape
# _foggy_beta_0.005.png #### LESS FOG 
# _foggy_beta_0.01.png #### MEDIUM FOG 
# _foggy_beta_0.02.png ##### DENSE FOG 

# LEVEL=(0.005 0.01 0.02)
# for FOG in "${LEVEL[@]}"
# do
#   printf "\n\n #### FoggyCitiscape (Amodal) FOG - Level : $FOG \n\n" >> $OUTPUT
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#     DATASETS.TEST '("FoggyCityscape_amodal_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.FOG_LEVEL $FOG >> $OUTPUT

#   printf "\n\n #### FoggyCitiscape (modal) FOG - Level : $FOG  \n\n" >> $OUTPUT
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#     DATASETS.TEST '("FoggyCityscape_modal_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.FOG_LEVEL $FOG >> $OUTPUT
# done








  


# # ###### Low-res2
# MODE="low-res2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE >> $OUTPUT

  
  





# ###### LVIS  
# OUTPUT=$OUTPUT-LVIS 
# printf "\n\n #### OG LVIS \n\n" >> $OUTPUT
# printf "\n\n MINi LVIS \n\n" >> $OUTPUT
# NUM_GPU=1 
# BATCHSIZE=1 #### FIX IT = 1 dont do 4,5 only works for 1
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   --task_config configs/lvis/minival.yaml \
#   TEST.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) SOLVER.IMS_PER_BATCH $NUM_GPU \
#   TEST.EVAL_TASK detection TEST.CHUNKED_EVALUATION 40 \
#   TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300 \
#   MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 \
#   OUTPUT_DIR TEMP  >>  $OUTPUT
 
  

# ###### ODINW-13
# OUTPUT_ODIN=$OUTPUT-ODWIN-13 
# ODIN_13=configs/odinw_13
# for TASK_CONFIG in "$ODIN_13"/*.yaml
# do
#   printf "\n\n #### $TASK_CONFIG \n\n" >> $OUTPUT_ODIN
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#           --task_config $TASK_CONFIG \
#           TEST.IMS_PER_BATCH $NUM_GPU SOLVER.IMS_PER_BATCH $NUM_GPU \
#           TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
#           DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False DATASETS.USE_OVERRIDE_CATEGORY True \
#           DATASETS.USE_CAPTION_PROMPT True \
#           OUTPUT_DIR TEMP >> $OUTPUT_ODIN
# done
# python Script/avg_odin2.py $OUTPUT_ODIN 13 >> $OUTPUT_ODIN



# ###### ODINW-13 + LR
# OUTPUT_ODIN=$OUTPUT-ODWIN-13_NOISE 
# ODIN_13=configs/odinw_13
# MODE="low-res"
# DATALOADER=COCODataset_Perturb
# for TASK_CONFIG in "$ODIN_13"/*.yaml
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   printf "\n\n #### $TASK_CONFIG \n\n" >> $OUTPUT_ODIN
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#           --task_config $TASK_CONFIG \
#           TEST.IMS_PER_BATCH $NUM_GPU SOLVER.IMS_PER_BATCH $NUM_GPU \
#           TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
#           DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False DATASETS.USE_OVERRIDE_CATEGORY True \
#           DATASETS.USE_CAPTION_PROMPT True \
#           DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
#           OUTPUT_DIR TEMP >> $OUTPUT_ODIN
# done
# python Script/avg_odin2.py $OUTPUT_ODIN 13 >> $OUTPUT_ODIN




    


# ###### ODINW-35
# OUTPUT_ODIN=$OUTPUT-ODWIN-35 
# ODIN_35=configs/odinw_35
# for TASK_CONFIG in "$ODIN_35"/*.yaml
# do
#   if [[ $TASK_CONFIG == "configs/odinw_35/plantdoc_100x100.yaml" ]]; then
#       continue 
#   fi
#   printf "\n\n #### $TASK_CONFIG \n\n" >> $OUTPUT_ODIN
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#           --task_config $TASK_CONFIG \
#           TEST.IMS_PER_BATCH $NUM_GPU SOLVER.IMS_PER_BATCH $NUM_GPU \
#           TEST.EVAL_TASK detection DATASETS.TRAIN_DATASETNAME_SUFFIX _grounding \
#           DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE False DATASETS.USE_OVERRIDE_CATEGORY True \
#           DATASETS.USE_CAPTION_PROMPT True \
#           OUTPUT_DIR TEMP >> $OUTPUT_ODIN
# done
# python Script/avg_odin2.py $OUTPUT_ODIN 36 >> $OUTPUT_ODIN




# # ###### FLICKER
# OUTPUT_FLK=$OUTPUT-FLICKER
# TASK_CONFIG=configs/flickr/test.yaml
# printf "\n\n #### OG FLICKER \n\n" >> $OUTPUT_FLK
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   --task_config $TASK_CONFIG OUTPUT_DIR TEMP \
#   SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 100 TEST.EVAL_TASK grounding MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False >> $OUTPUT_FLK




# #################### Virtual Kitty 
# OUTPUT_KITTI=$OUTPUT-KITTI-VAL
# DATASET_NAME='("kitti_test",)'
# DATASET_NAME='("kitti_debug",)'
# DATASET_NAME='("kitti_val",)'
# printf "\n\n #### Virtual Kitti  \n\n" >> $OUTPUT_KITTI
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST $DATASET_NAME MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False >> $OUTPUT_KITTI
  
# :::::: Results :
# 15-deg-left          ::   AP  : 0.212358                AP50: 0.383585          AP75: 0.212561          APs : 0.200865          APm : 0.172849          APl : 0.174136
# 15-deg-right         ::   AP  : 0.200865                AP50: 0.372046          AP75: 0.194379          APs : 0.172849          APm : 0.174136          APl : 0.159986
# 30-deg-left          ::   AP  : 0.172849                AP50: 0.319813          AP75: 0.172373          APs : 0.174136          APm : 0.159986          APl : 0.152997
# 30-deg-right         ::   AP  : 0.174136                AP50: 0.320237          AP75: 0.168588          APs : 0.159986          APm : 0.152997          APl : 0.119032
# clone                ::   AP  : 0.159986                AP50: 0.297305          AP75: 0.155979          APs : 0.152997          APm : 0.119032          APl : 0.115945
# fog                  ::   AP  : 0.152997                AP50: 0.291776          AP75: 0.148036          APs : 0.119032          APm : 0.115945          APl : 0.113528
# morning              ::   AP  : 0.119032                AP50: 0.231666          AP75: 0.111098          APs : 0.115945          APm : 0.113528          APl : 0.116475
# overcast             ::   AP  : 0.115945                AP50: 0.235279          AP75: 0.105060          APs : 0.113528          APm : 0.116475          APl : 0.070322
# rain                 ::   AP  : 0.113528                AP50: 0.221651          AP75: 0.105503          APs : 0.116475          APm : 0.070322          APl : 0.057616
# sunset               ::   AP  : 0.116475                AP50: 0.235872          AP75: 0.104781          APs : 0.070322          APm : 0.057616          APl : 0.206966
# Overall              ::   AP  : 0.224518                AP50: 0.377343          AP75: 0.265937          APs : 0.026365          APm : 0.280485          APl : 0.505097




#################### WIDER FACE 
OUTPUT_FACE=$OUTPUT-WIDERFACE-VAL
DATASET_NAME='("wider_face-debug",)'
DATASET_NAME='("wider_face-val",)'
printf "\n\n #### WIDER FACE \n\n" >> $OUTPUT_FACE
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST $DATASET_NAME MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False >> $OUTPUT_FACE
  
    


# #################### VisDRONE-2019
OUTPUT_FACE=$OUTPUT-VIS-VAL
DATASET_NAME='("vis_dron2019-test",)'
DATASET_NAME='("vis_dron2019-val",)'
DATASET_NAME='("vis_dron2019-test-easy",)'
printf "\n\n #### VIS DRONE \n\n" >> $OUTPUT_FACE
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST $DATASET_NAME MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False >> $OUTPUT_FACE
  

# # #################### UAVDT
# OUTPUT_FACE=$OUTPUT-UAVDT-VAL
# DATASET_NAME='("UAVDT-debug",)'
# printf "\n\n #### UAVDT \n\n" >> $OUTPUT_FACE
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST $DATASET_NAME MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False >> $OUTPUT_FACE
  
        












  
  
  
  


# rsync -r /data/priyank/synthetic/leftImg8bit_foggyDBF ucf0:/home/c3-0/datasets/
# /home/c3-0/datasets/leftImg8bit_foggyDBF

rsync -a ucf_output/* ucf2:~/robustness_object_detection/GLIP/ucf_output/
# rsync -a ucf0:~/robustness_object_detection/GLIP/ucf_output/* ~/robustness_object_detection/GLIP/ucf_output/



# cd ~/robustness_object_detection/GLIP/
# sbatch Script/vanilla_benchmark.sh