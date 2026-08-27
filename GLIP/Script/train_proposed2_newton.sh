#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=OR_N_15
#SBATCH --output=ucf_output_newton/slurm-%j.out
#SBATCH --constraint=gpu80

#######SBATCH --time=3:00:00
######SBATCH --partition=preemptable --qos preemptable
#####SBATCH -C '!gmem16'
###### srun --pty -t24:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu16 bash
###### srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu32 bash
###### srun --pty -t24:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu32 bash
###### srun --pty -t180:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu32 bash
###### srun --pty -t200:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu80 bash
###### srun --pty -t1:00:00 --gres=gpu:1 --cpus-per-gpu=6 --constraint=gpu80 bash
# srun --pty -t1:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu80 bash
# srun --pty -t1:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu80 --partition=preemptable --qos preemptable bash
# srun --pty -t1:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu80 --partition=preemptable --qos preemptable bash
# srun --pty -t1:00:00 bash
# srun --pty -t1:00:00 --gres=gpu:1 --cpus-per-gpu=6 --constraint=gpu32 bash

nvidia-smi
source ~/.bashrc
source ~/.bash_profile

source activate OD 
conda activate OD 
module load cuda/cuda-11.3
module load cuda/cuda-11.8
module load gcc/gcc-9.1.0


echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
scontrol write batch_script $SLURM_JOB_ID
mv slurm-$SLURM_JOB_ID.sh ucf_output_newton/
rsync -a ucf_output_newton/slurm-$SLURM_JOB_ID.sh ucf2:~/robustness_object_detection/GLIP/ucf_output_newton/

cd ~/robustness_object_detection/GLIP/


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

export DETECTRON2_DATASETS=/datasets/flickr_dataset_30k/
export DATASET=/datasets/flickr_dataset_30k/
export ADD_DATASET=./
export TOKENIZERS_PARALLELISM=true 

NUM_GPU=2


# ln -s /datasets/MSCOCO17/* coco/ 


###################### GLIP-T [5] ########################################################################################
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output_newton/GLIP-T_5.txt
MODEL_TYPE=GLIP-T_5

# CONFIG=configs/pretrain/glip_Swin_L.yaml
# WT=MODEL/glip_large_model.pth
# OUTPUT=ucf_output_newton/GLIP-L.txt
# MODEL_TYPE=GLIP-L



###################### LR-TKO
BACKBONE_TYPE=LR_TKO
BATCHSIZE=3

# ###################### LR-TKO (SR)
# BACKBONE_TYPE=LR_TKO_SR
# BATCHSIZE=3

# ###################### LoRA
# BACKBONE_TYPE=LORA
# BATCHSIZE=3

# ###################### LoRA
# BACKBONE_TYPE=LORA_SR
# BATCHSIZE=3


# ###################### Adapters 
# BACKBONE_TYPE=ADAPTER
# BATCHSIZE=3

# ###################### Adapters 
# BACKBONE_TYPE=ADAPTER_SR
# BATCHSIZE=3


# ###################### VPT
# BACKBONE_TYPE=VPT_DEEP
# BATCHSIZE=3

# ###################### VPT
# BACKBONE_TYPE=VPT_DEEP_SR
# BATCHSIZE=3




TECHNIQUE=GeneralizedVLRCNN_TRILANG
HEAD=VLDyHeadModule_LANG_TRI


# ###### FLICKER TRAINING & COCO EVAL 
TRAIN_FN=multi_scale_train_n_captions
DATALOADER=FlickrDataset_MULTI_SCALE_LRCAPT
TEST_DATALOADER=COCODataset_Perturb
SUBSET=subset3
SUFFIX=FL









####### ######## ######## ######## ######## ######## ######## #######
####### ######## ######## ######## ######## ######## ######## #######


####### ######## ######## ######## 
# Baseline Training 
####### ######## ######## ######## 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# BATCHSIZE=3
# NORMAL_NOISES=("snow_model2" "rain_model2")
# for LOSS_MODE in "${NORMAL_NOISES[@]}"
# do
#   DUMP_FOLDER=$ROOT_DUMP_FOLDER+$LOSS_MODE/
#   rm -rf $DUMP_FOLDER

#   OUTPUT_NOISE=$OUTPUT-$LOSS_MODE
#   printf "\n\n #### $LOSS_MODE \n\n" >> $OUTPUT_NOISE
#   CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 1 >> $OUTPUT_NOISE
# done


####### ######## ######## ######## 
# MSE loss on feats (naive)
####### ######## ######## ######## 
# BATCHSIZE=3
# EXP_NAME=MSE
# DATALOADER=FlickrDataset_MULTI_SCALE
# TRAIN_FN=multi_scale_train

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 2 MODEL.RPN.NO_FUSE mse >> $OUTPUT
    

####### ######## ######## ######## 
# Langugage based triplet loss 
####### ######## ######## ######## 
# BATCHSIZE=2
# N_LR=2
# EXP_NAME=Lang-Tri-$N_LR

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_LR $N_LR >> $OUTPUT



# # ####### ######## ######## ######## ######## ######## ######## #######
# # # Neg based (Triplet on negative samples mostly)
# # ####### ######## ######## ######## ######## ######## ######## #######
# N_LR=2
# BATCHSIZE=2
# EXP_NAME=Neg_Lang_Tri-$N_LR

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     MODEL.RPN.NO_FUSE one_anchor DATASETS.N_LR $N_LR >> $OUTPUT


# # ####### ######## ######## ######## ######## ######## ######## #######
# # # Neg based (Triplet loss maximizing distance with dirtyfused emebddings)
# # ####### ######## ######## ######## ######## ######## ######## #######
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# BATCHSIZE=3
# EXP_NAME=DiFu-Tri

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 2 MODEL.RPN.NO_FUSE neg_triple >> $OUTPUT

# --spawn-method  
# tensor([  1,   1,   2,   2,   3,   3, 101, 101, 102, 102, 103, 103,   4,   4, 5,   5,   6,   6, 104, 104, 105, 105, 106, 106],


####### ######## ######## ######## 
# Triplet loss on 2 NO REPLICA (naive)
####### ######## ######## ######## 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# EXP_NAME=Tri
# BATCHSIZE=3

# LOSS_MODE="low-res"
# # LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 2 MODEL.RPN.NO_FUSE replica >> $OUTPUT
    
# --spawn-method 


###### ######## ######## ######## 
# Distillation loss on fused feats (no new captions)
###### ######## ######## ######## 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# TECHNIQUE=GeneralizedVLRCNN_DISTILL
# HEAD=VLDyHeadModule_LANG_Distill
# EXP_NAME=Distill
# BATCHSIZE=3

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD DATASETS.N_BR 2 MODEL.RPN.NO_FUSE distill >> $OUTPUT
    

# ####### ######## ######## ######## 
# # GNoise on Feats
# ####### ######## ######## ######## 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# EXP_NAME=GNoise
# BATCHSIZE=3

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 2 MODEL.RPN.NO_FUSE noise >> $OUTPUT


# # ####### ######## ######## ######## 
# # # Multi Scale Augmentations variants 
# # ####### ######## ######## ######## 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE_MULTI_LR
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# EXP_NAME=MS-Aug
# BATCHSIZE=2

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 6 MODEL.RPN.NO_FUSE replica >> $OUTPUT
# --spawn-method 


    
    














####### ######## ######## ######## ######## ######## ######## #######
######## MULTIPLE NOISES 
####### ######## ######## ######## ######## ######## ######## #######

# DATALOADER=FlickrDataset_MULTI_SCALE_LRCAPT_MULTI_AUG
# NORMAL_NOISES=(motion_blur2 low-res jpg_compression fog focus_blur)
# N_NAME=MB_LR_J_FG_F
# TEST_NOISE=jpg_compression  

# NORMAL_NOISES=(snow_model2 rain_model2 iso_blur atmospheric)
# N_NAME=S_R_IS_AT
# TEST_NOISE=iso_blur  

# NORMAL_NOISES=(salt_pepper pixel_dropout)
# N_NAME=SP_PD
# TEST_NOISE=salt_pepper  


# LOSS_MODE=$(IFS="*" ; echo "${NORMAL_NOISES[*]}")
# echo $LOSS_MODE



####### ######## ######## ######## 
# Langugage based triplet loss 
####### ######## ######## ######## 
# N_LR=1
# EXP_NAME=MN-Lang_Tri-$N_LR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS_NEWTON/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.ONE_NOISE True DATASETS.TEST_NOISE $TEST_NOISE \
#     DATASETS.N_LR $N_LR >> $OUTPUT
    
#  --spawn-method

####### ######## ######## ######## 
# Vanilla Training (simple Augmentation)
####### ######## ######## ######## 

# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# TRAIN_FN=multi_scale_train
# N_LR=1 #### Doesnt matter 
# EXP_NAME=MN-Aug-$N_LR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
    # --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
    # SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
    # DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
    # MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATASETS.ONE_NOISE True DATASETS.TEST_NOISE $TEST_NOISE \
    # DATASETS.N_LR $N_LR >> $OUTPUT
    


####### ######## ######## ######## 
# Multi-Scale Augmentation 
####### ######## ######## ######## 
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE_MULTI_LR
# DATALOADER=FlickrDataset_MULTI_SCALE_LRCAPT_MULTI_AUG
# EXP_NAME=AN
# BATCHSIZE=2

# LOSS_MODE="low-res"
# N_LR=1 #### Doesnt matter 

# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME

# DUMP_FOLDER=$ROOT_DUMP_FOLDER+$LOSS_MODE+$EXP_NAME/
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 6 MODEL.RPN.NO_FUSE replica >> $OUTPUT


####### ######## ######## ######## 
# MSE Loss 
####### ######## ######## ######## 
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# TRAIN_FN=multi_scale_train
# N_LR=1 #### Doesnt matter 
# BATCHSIZE=3
# EXP_NAME=MSE-$N_LR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output_newton/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.ONE_NOISE True DATASETS.TEST_NOISE $TEST_NOISE \
#     DATASETS.N_BR 2 DATASETS.N_LR $N_LR MODEL.RPN.NO_FUSE mse >> $OUTPUT
    














rsync -a --exclude='*/*' ~/robustness_object_detection/GLIP/ucf_output_newton/* ucf2:~/robustness_object_detection/GLIP/ucf_output_newton/
rsync -a --exclude='*/*' ~/robustness_object_detection/GLIP/ucf_output/* ucf2:~/robustness_object_detection/GLIP/ucf_output/
rsync -r ~/robustness_object_detection/GLIP/EXPS_NEWTON/ ucf0:~/robustness_object_detection/GLIP/EXPS_NEWTON/
rsync -r ~/robustness_object_detection/GLIP/EXPS/ ucf0:~/robustness_object_detection/GLIP/EXPS_NEWTON/




# rsync -a --exclude='*/*' ucf4:~/robustness_object_detection/GLIP/ucf_output_newton/* ~/robustness_object_detection/GLIP/ucf_output_newton/

# rsync -r EXPS_NEWTON ucf0:~/robustness_object_detection/GLIP/




# cd ~/robustness_object_detection/GLIP/
# sbatch Script/train_proposed2_newton.sh


