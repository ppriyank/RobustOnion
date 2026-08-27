#!/bin/bash

#SBATCH --job-name=OR_11
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH --exclude=c4-4,c5-2
#SBATCH -C gmem48 --gres=gpu:2 --mem-per-cpu=6G -c8

################SBATCH -p gpu --qos=preempt
################SBATCH -p gpu --qos=day
#################SBATCH -p gpu --qos=short 

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

NUM_GPU=2


if [[ "$SLURM_JOB_NODELIST" == "c4-4" ]]; then
        sbatch ucf_output/slurm-$SLURM_JOB_ID.sh
        exit 1
fi





###################### GLIP-T [5] ########################################################################################
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output/GLIP-T_5.txt
MODEL_TYPE=GLIP-T_5


###################### LR-TKO
# BACKBONE_TYPE=LR_TKO
# BATCHSIZE=2

# ###################### LoRA
# BACKBONE_TYPE=LORA
# BATCHSIZE=2

###################### LR-TKO (SR)
BACKBONE_TYPE=LR_TKO_SR
BATCHSIZE=2

# ###################### LoRA
# BACKBONE_TYPE=LORA_SR
# BATCHSIZE=2


# ###################### Adapters 
# BACKBONE_TYPE=ADAPTER
# BATCHSIZE=3

# # ###################### VPT
# # BACKBONE_TYPE=VPT_DEEP
# # BATCHSIZE=4


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
# Langugage based triplet loss 
####### ######## ######## ######## 
# N_LR=2
# EXP_NAME=Lang-Tri-$N_LR

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"

# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_LR $N_LR >> $OUTPUT
    

# # ######## Virtual Kitty
TRAIN_FN=multi_scale_train
Dataset_TRAIN='("kitti_train-grounding",)'
Dataset_TEST='("kitti_val",)'
SUFFIX=VK
echo "External dataset ..... "
BATCHSIZE=4

N_LR=4
EXP_NAME=Lang-Tri-$N_LR
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
OUTPUT=ucf_output/$NAME.txt
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
    --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
    SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
    DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
    MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
    DATASETS.N_LR $N_LR DATASETS.N_BR 1 >> $OUTPUT
    


####### ######## ######## ######## 
# Triplet loss on 2 NO REPLICA (naive)
####### ######## ######## ######## 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# EXP_NAME=Tri-1
# BATCHSIZE=3

# LOSS_MODE="low-res"
# LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 2 MODEL.RPN.NO_FUSE replica >> $OUTPUT
    

####### ######## ######## ######## 
# MSE loss on feats (naive)
####### ######## ######## ######## 
# N_LR=2
# BATCHSIZE=3
# EXP_NAME=MSE-$N_LR
# DATALOADER=FlickrDataset_MULTI_SCALE
# TRAIN_FN=multi_scale_train

# LOSS_MODE="low-res"
# # LOSS_MODE="atmospheric"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$LOSS_MODE-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.N_BR 2 MODEL.RPN.NO_FUSE mse >> $OUTPUT
    
# tensor([1, 1, 2, 2, 3, 3], device='cuda:0')                                                                                                                                          
# tensor([4, 4, 5, 5, 6, 6], device='cuda:1') 




####### ######## ######## ######## ######## ######## ######## #######
######## MULTIPLE NOISES 
####### ######## ######## ######## ######## ######## ######## #######

DATALOADER=FlickrDataset_MULTI_SCALE_LRCAPT_MULTI_AUG
NORMAL_NOISES=(motion_blur2 low-res jpg_compression fog focus_blur)
N_NAME=MB_LR_J_FG_F
TEST_NOISE=jpg_compression  

NORMAL_NOISES=(snow_model2 rain_model2 iso_blur atmospheric)
N_NAME=S_R_IS_AT
TEST_NOISE=iso_blur  

NORMAL_NOISES=(salt_pepper pixel_dropout)
N_NAME=SP_PD
TEST_NOISE=salt_pepper  


LOSS_MODE=$(IFS="*" ; echo "${NORMAL_NOISES[*]}")
echo $LOSS_MODE


####### ######## ######## ######## 
# Langugage based triplet loss 
####### ######## ######## ######## 
# N_LR=2
# EXP_NAME=MN-Lang_Tri-$N_LR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.ONE_NOISE True DATASETS.TEST_NOISE $TEST_NOISE \
#     DATASETS.N_LR $N_LR >> $OUTPUT
    
####### ######## ######## ######## 
# Vanilla Training (simple Augmentation)
####### ######## ######## ######## 
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# TRAIN_FN=multi_scale_train
# N_LR=1 #### Doesnt matter 
# EXP_NAME=MN-Aug-$N_LR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATASETS.ONE_NOISE True DATASETS.TEST_NOISE $TEST_NOISE \
#     DATASETS.N_LR $N_LR >> $OUTPUT
    

####### ######## ######## ######## 
# MSE Loss 
####### ######## ######## ######## 
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# TRAIN_FN=multi_scale_train
# N_LR=1 #### Doesnt matter 
# BATCHSIZE=3
# EXP_NAME=MSE-$N_LR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$N_NAME-$EXP_NAME-$SUFFIX
# OUTPUT=ucf_output/$NAME.txt
# DUMP_FOLDER=EXPS/$NAME
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --spawn-method \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD \
#     DATASETS.ONE_NOISE True DATASETS.TEST_NOISE $TEST_NOISE \
#     DATASETS.N_BR 2 DATASETS.N_LR $N_LR MODEL.RPN.NO_FUSE mse >> $OUTPUT
    

# --spawn-method \
# --display_images "all" 
# DATASETS.ONE_NOISE True augment only 1 noise at a time 












rsync -a --exclude='*/*' ucf_output/* ucf2:~/robustness_object_detection/GLIP/ucf_output/
# rsync -a --exclude='*/*' ucf0:~/robustness_object_detection/GLIP/ucf_output/* ~/robustness_object_detection/GLIP/ucf_output/



# cd ~/robustness_object_detection/GLIP/
# sbatch Script/train_proposed2.sh


