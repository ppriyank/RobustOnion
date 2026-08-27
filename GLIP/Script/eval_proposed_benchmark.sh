#!/bin/bash

#SBATCH --job-name=OR_1
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G --cpus-per-gpu=8
#SBATCH --exclude=c4-4,c5-2

################SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G -c8


################SBATCH -p gpu --qos=day
################SBATCH -p gpu --qos=short 
###############SBATCH -p gpu --qos=preempt 

# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --qos preempt bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash 
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
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
MODEL_TYPE=GLIP-T_5
DATALOADER=COCODataset_Perturb




# BACKBONE_TYPE=LR_TKO
# LOSS_MODE=low-res
# LOSS_MODE=jpg_compression
# LOSS_MODE=pixel_dropout
# LOSS_MODE=fog
# LOSS_MODE=iso_blur
# LOSS_MODE=atmospheric
# LOSS_MODE=rain_model2
# LOSS_MODE=snow_model2
# LOSS_MODE=focus_blur
# LOSS_MODE=motion_blur2
# LOSS_MODE=salt_pepper
# LOSS_MODE=chromatic

# BACKBONE_TYPE=VPT_DEEP
# LOSS_MODE=iso_blur
# LOSS_MODE=pixel_dropout
# LOSS_MODE=salt_pepper
# LOSS_MODE=low-res
# LOSS_MODE=fog
# LOSS_MODE=motion_blur2
# LOSS_MODE=chromatic
# LOSS_MODE=focus_blur
# LOSS_MODE=jpg_compression
# LOSS_MODE=snow_model2
# LOSS_MODE=rain_model2
# LOSS_MODE=atmospheric


# BACKBONE_TYPE=LORA
# LOSS_MODE=jpg_compression
# LOSS_MODE=pixel_dropout
# LOSS_MODE=low-res
# LOSS_MODE=salt_pepper
# LOSS_MODE=iso_blur
# LOSS_MODE=fog
# LOSS_MODE=motion_blur2
# LOSS_MODE=chromatic
# LOSS_MODE=focus_blur
# LOSS_MODE=atmospheric
# LOSS_MODE=rain_model2
# LOSS_MODE=snow_model2

# BACKBONE_TYPE=ADAPTER
# LOSS_MODE=pixel_dropout
# LOSS_MODE=atmospheric
# LOSS_MODE=snow_model2
# LOSS_MODE=rain_model2
# LOSS_MODE=fog
# LOSS_MODE=salt_pepper
# LOSS_MODE=motion_blur2
# LOSS_MODE=chromatic
# LOSS_MODE=iso_blur
# LOSS_MODE=jpg_compression
# LOSS_MODE=low-res
# LOSS_MODE=focus_blur


# DUMP_FOLDER=EXPS/$MODEL_TYPE-$BACKBONE_TYPE
# DUMP_FOLDER=$DUMP_FOLDER+$LOSS_MODE
# OUTPUT=ucf_output/$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-EVAL.txt
# WT=$DUMP_FOLDER/model_best.pth
# ls $WT 

##############################################
####################### PROPOSED 
##############################################

# EXP_FOLDER=EXPS_NEWTON
# SUFFIX=FL

# BACKBONE_TYPE=LR_TKO
# BACKBONE_TYPE=LR_TKO_SR
# BACKBONE_TYPE=LORA
# BACKBONE_TYPE=LORA_SR
# BACKBONE_TYPE=VPT_DEEP_SR
# BACKBONE_TYPE=VPT_DEEP
# BACKBONE_TYPE=ADAPTER_SR
# BACKBONE_TYPE=ADAPTER


# EXP_FOLDER=EXPS
# ###### vanilla  
# BACKBONE_TYPE=VANIL
# # EXP_NAME=E2E
# EXP_NAME=Fuse
# SUFFIX=BDD-ALL
# SUFFIX=BDD-ALL-L2

##### Cross exchange non-local 
# BACKBONE_TYPE=NN
# BACKBONE_TYPE=NN+LR_TKO
# BACKBONE_TYPE=NN+Lin
# BACKBONE_TYPE=NN+Lin+Norm
# BACKBONE_TYPE=NN+Lin+Norm+LR_TKO
# BACKBONE_TYPE=NN+Lin+LR_TKO
# BACKBONE_TYPE=NN+Norm+LR_TKO
# BACKBONE_TYPE=NN+Norm
# BACKBONE_TYPE=NN+Norm






# EXP_NAME=MSE
# EXP_NAME=GNoise
# EXP_NAME=Tri
# EXP_NAME=DiFu-Tri
# EXP_NAME=Neg_Lang_Tri-2
# EXP_NAME=Distill
# EXP_NAME=Lang-Tri-3
# EXP_NAME=Lang-Tri-2
# EXP_NAME=Tri-1
# EXP_NAME=MS-Aug

# EXP_NAME=MN-Lang_Tri-2
# EXP_NAME=MSE-1
# EXP_NAME=MN-Lang_Tri-1


# LOSS_MODE="low-res"
# LOSS_MODE=atmospheric
# LOSS_MODE=MB_LR_J_FG_F
# LOSS_MODE=SP_PD
# LOSS_MODE=S_R_IS_AT
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$EXP_NAME-$SUFFIX
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME
# NAME=$MODEL_TYPE-$BACKBONE_TYPE+$LOSS_MODE
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX


# OUTPUT=ucf_output/$NAME-EVAL.txt
# DUMP_FOLDER=$EXP_FOLDER/$NAME
# WT=$DUMP_FOLDER/model_best.pth
# ls $WT


# GLIP/ucf_output/GLIP-T_5-VANIL-Fuse-BDD-ALL-L2-EVAL.txt

# GLIP/ucf_output/GLIP-T_5-VANIL-E2E-BDD-ALL.txt

# GLIP/ucf_output/GLIP-T_5-VANIL-Fuse-BDD-ALL.txt
# GLIP/ucf_output/GLIP-T_5-VANIL-Fuse-BDD-ALL-L2.txt

# GLIP/ucf_output/GLIP-T_5-LR_TKO-BDD-ALL-L2.txt
# GLIP/ucf_output/GLIP-T_5-LR_TKO-BDD-ALL.txt

# GLIP/ucf_output/GLIP-T_5-NN+Norm-BDD-ALL-L2.txt

# GLIP/ucf_output/GLIP-T_5-NN+Norm+LR_TKO-BDD-ALL-L2.txt
# GLIP/ucf_output/GLIP-T_5-NN+Norm+LR_TKO-BDD-ALL.txt





###### COCO 
# printf "\n\n #### OG \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT


# ##################### Noise  - 1
# NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog" "low-res2")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   ####### Vanilla Training 
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT
# done

# NOISES=("low-res" "motion_blur2")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   ####### Vanilla Training 
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT
# done



###################### Noise  - 2
# NOISES=("rain_model2" "snow_model2" "atmospheric")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   ####### Vanilla Training 
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATALOADER.NUM_WORKERS 0 \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT
# done

# NOISES=("atmospheric")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   ####### Vanilla Training 
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATALOADER.NUM_WORKERS 0 \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT
# done



################################################################################################

# SUFFIX=VK-FT
# SUFFIX=VK-L2
# SUFFIX=VK-L3
# SUFFIX=-Lang-Tri--VK
# SUFFIX=VK-E2E-1
# SUFFIX=VK-E2E-2
# SUFFIX=VK-E2E-L3-2
# SUFFIX=VK-E2E-L3
# SUFFIX=--VK
# SUFFIX=VK
# SUFFIX=BDD

# SUFFIX=VK-E2E-FT-2
# SUFFIX=VK-E2E-FT-3
# SUFFIX=VK-E2E-FT-4
# SUFFIX=VK-E2E-FT



# ################################# Fine-tuning Results 
# BACKBONE_TYPE=LR_TKO
# BACKBONE_TYPE=LORA
# BACKBONE_TYPE=ADAPTER
# BACKBONE_TYPE=LR_TKO_SR
# BACKBONE_TYPE=VPT_DEEP

# OUTPUT=ucf_output/$MODEL_TYPE-$BACKBONE_TYPE-FINE-T-EVAL.txt
# ROOT_DUMP_FOLDER=EXPS/$MODEL_TYPE-$BACKBONE_TYPE
# DUMP_FOLDER=$ROOT_DUMP_FOLDER+WEDGE/
# WT=$DUMP_FOLDER/model_best.pth
# ls $WT 


# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# ###### Only channel attention Train 
# BACKBONE_TYPE=FAN
###### Entire MLP + Channel Attention 
# BACKBONE_TYPE=FAN_MLP
###### Entire Network E2E 
# BACKBONE_TYPE=FAN_E2E

# ###### Baseline entire Kitty Registers Extra Class tokens / Registers 
# BACKBONE_TYPE=REGISTER
# BACKBONE_TYPE=REGISTER_E2E

# # ###### Mint :: Only Layer Norm 
# BACKBONE_TYPE=MINT
# TECHNIQUE=GeneralizedVLRCNN_VAR

# BACKBONE_TYPE=LORA
# BACKBONE_TYPE=LORA_SR
# BACKBONE_TYPE=LR_TKO
# BACKBONE_TYPE=LR_TKO_SR

# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX


# LOSS_MODE="15-deg-left"
# LOSS_MODE="15-deg-right"
# LOSS_MODE="30-deg-left"
# LOSS_MODE="30-deg-right"
# LOSS_MODE="clone"
# LOSS_MODE="fog"
# LOSS_MODE="morning"
# LOSS_MODE="overcast"
# LOSS_MODE="rain"
# LOSS_MODE="sunset"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE+$LOSS_MODE+$SUFFIX





# OUTPUT=ucf_output/$NAME-EVAL.txt
# DUMP_FOLDER=EXPS/$NAME
# WT=$DUMP_FOLDER/model_best.pth
# ls $WT 

# # ###### BBDK100
# printf "\n\n #### BBDK \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("bdd100k_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT


# # ###### DAWN
# printf "\n\n #### DAWN \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("dawn_test",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT


# # ###### WEDGE
# printf "\n\n #### WEDGE \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("wedge_test",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT


################## FoggyCitiscape
# _foggy_beta_0.005.png #### LESS FOG 
# _foggy_beta_0.01.png #### MEDIUM FOG 
# _foggy_beta_0.02.png ##### DENSE FOG 

# LEVEL=(0.005 0.01 0.02)
# # LEVEL=(0.01)
# for FOG in "${LEVEL[@]}"
# do
#   printf "\n\n #### FoggyCitiscape (Amodal) FOG - Level : $FOG \n\n" >> $OUTPUT
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#     DATASETS.TEST '("FoggyCityscape_amodal_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE \
#     DATASETS.FOG_LEVEL $FOG >> $OUTPUT

#   printf "\n\n #### FoggyCitiscape (modal) FOG - Level : $FOG  \n\n" >> $OUTPUT
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#     OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#     DATASETS.TEST '("FoggyCityscape_modal_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE \
#     DATASETS.FOG_LEVEL $FOG >> $OUTPUT
# done



# ###### Virtual Kitti 
# printf "\n\n #### Virtual Kitti \n\n" >> $OUTPUT
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
#   OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
#   DATASETS.TEST '("kitti_test",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE >> $OUTPUT  

  
  













rsync -a --exclude='*/*' ucf_output/* ucf2:~/robustness_object_detection/GLIP/ucf_output/
# rsync -a --exclude='*/*' ucf0:~/robustness_object_detection/GLIP/ucf_output/* ~/robustness_object_detection/GLIP/ucf_output/



# cd ~/robustness_object_detection/GLIP/
# sbatch Script/eval_proposed_benchmark.sh


