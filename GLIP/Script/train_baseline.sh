#!/bin/bash

#SBATCH --job-name=B_7
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH --exclude=c4-4,c5-2
#SBATCH -C gmem48 --gres=gpu:2 --mem-per-cpu=6G 
#SBATCH --cpus-per-gpu=8

####SBATCH -c8
############## SBATCH -c8
################SBATCH -p gpu --qos=preempt
##################SBATCH -p gpu --qos=day
#################SBATCH -p gpu --qos=short 

# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --qos preempt bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash 
# srun --pty --gres=gpu:2 --cpus-per-gpu=8 -C gmem48 --exclude=c4-4,c5-2 bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 --qos preempt bash 
# srun --pty --cpus-per-task=8 bash 

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
OUTPUT=ucf_output/GLIP-T_5
MODEL_TYPE=GLIP-T_5

TRAIN_FN=multi_scale_train
TECHNIQUE=GeneralizedVLRCNN_PROPOSE


# # ###### WEDGE
# SUFFIX=WEDGE
# Dataset='("wedge_test",)'
# Dataset='("wedge_train-grounding",)'
# BATCHSIZE=10

# ###### Virtual Kitty
# Dataset_TRAIN='("kitti_train-grounding-debug",)'
# Dataset_TRAIN='("kitti_train-grounding",)'
# Dataset_TEST='("kitti_val",)'
# BATCHSIZE=15
# SUFFIX=VK

###### Virtual Kitty Pairs
# Dataset_TRAIN='("kitti_Grounding_Pairs-grounding",)'
# Dataset_TEST='("kitti_val",)'
# BATCHSIZE=15
# SUFFIX=VK-P

# ###### BDDK
# Dataset_TRAIN='("bdd100k_train-grounding",)'
# Dataset_TEST='("bdd100k_val2",)'
# BATCHSIZE=1
# SUFFIX=BDD

# # ###### BDDK (Entire Training Set)
# Dataset_TRAIN='("bdd100k_all_train-grounding",)'
# Dataset_TEST='("bdd100k_val2",)'
# BATCHSIZE=2
# SUFFIX=BDD-ALL
# NUM_CLASSES=10
# echo "External dataset ..... "


# # ###### UAVDT (Entire Training Set)
# Dataset_TRAIN='("UAVDT-grounding",)'
# Dataset_TRAIN='("UAVDT-grounding2",)'
# Dataset_TEST='("vis_dron2019-val",)'
# BATCHSIZE=2
# SUFFIX=UAVDT-ALL
# NUM_CLASSES=4
# echo "External dataset ..... "

# rsync -r /data/priyank/synthetic/VisDrone2019-DET-val ucf0:/home/c3-0/datasets/
# rsync -a /data/priyank/synthetic/UAVDT/images.zip ucf0:/home/c3-0/datasets/UAVDT/
# rsync -r /data/priyank/synthetic/UAVDT/labels_coco ucf0:/home/c3-0/datasets/UAVDT/


# ##############################################################################################################################
# ############################################# Previous Work 



# ########################################## E2E  (no backbone)
# BACKBONE_TYPE=VANIL-E2E
# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#     MODEL.NO_NORM True >> $OUTPUT_NOISE.txt

# ############ E2E  (no backbone) Starting the evaluations at 4700
# BACKBONE_TYPE=VANIL-E2E
# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-L2
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#     MODEL.NO_NORM True SOLVER.USE_AMP True \
#     MODEL.DYHEAD.NUM_CLASSES 10 MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS True >> $OUTPUT_NOISE.txt
    

# ############# E2E  (no backbone) + LR noise 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# TEST_DATALOADER=COCODataset_Perturb
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# SUBSET=subset3

# BACKBONE_TYPE=VANIL-E2E
# BATCHSIZE=3
# LOSS_MODE="low-res"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-$LOSS_MODE
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX $LOSS_MODE \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET  \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#     MODEL.NO_NORM True SOLVER.USE_AMP True >> $OUTPUT_NOISE.txt






    
    
    




# ############# Fusion (no backbone) Starting the evaluations at 4700
# BACKBONE_TYPE=VANIL
# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-Fuse-$SUFFIX-L2
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
#   MODEL.E2E_MODE 'rpn_dyhead_tower' MODEL.E2E True \
#   MODEL.DYHEAD.NUM_CLASSES 10 MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS True >> $OUTPUT_NOISE.txt


# ############# Only Fusion Network & Frozen vanilla backbone
# BACKBONE_TYPE=VANIL
# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-Fuse-$SUFFIX
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#     --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#     SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#     DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#     MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE \
#     MODEL.E2E_MODE 'rpn_dyhead_tower' MODEL.E2E True >> $OUTPUT_NOISE.txt


############# Only Fusion Network & Frozen vanilla backbone + LR noise 
TRAIN_FN=multi_scale_train
DATALOADER=FlickrDataset_MULTI_SCALE
TEST_DATALOADER=COCODataset_Perturb
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
SUBSET=subset3

LOSS_MODE="low-res"
BACKBONE_TYPE=VANIL
BATCHSIZE=3
NAME=$MODEL_TYPE-$BACKBONE_TYPE-Fuse-$SUFFIX-$LOSS_MODE
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
rm -rf $DUMP_FOLDER
printf "\n\n #### $SUFFIX $LOSS_MODE \n\n" >> $OUTPUT_NOISE.txt
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
    --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET  \
    SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
    DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
    MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE \
    MODEL.E2E_MODE 'rpn_dyhead_tower' MODEL.E2E True >> $OUTPUT_NOISE.txt



    
    





# ########################################## LR-TKO / LoRA / VPT / Adapters
#### LoRA
# BACKBONE_TYPE=LORA
#### Adapters 
# BACKBONE_TYPE=ADAPTER
#### VPT
# BACKBONE_TYPE=VPT_DEEP
#### LR_TKO
# BACKBONE_TYPE=LR_TKO

# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True >> $OUTPUT_NOISE.txt



# ########################################## NN
# BACKBONE_TYPE=NN+Norm
# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True >> $OUTPUT_NOISE.txt


# ########################################## NN + LRTK0
# BACKBONE_TYPE=NN+Norm+LR_TKO
# BATCHSIZE=3
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True >> $OUTPUT_NOISE.txt


# ############# NN + LRTK0 + LR Noise 
# TRAIN_FN=multi_scale_train
# DATALOADER=FlickrDataset_MULTI_SCALE
# TEST_DATALOADER=COCODataset_Perturb
# TECHNIQUE=GeneralizedVLRCNN_PROPOSE
# SUBSET=subset3


# BACKBONE_TYPE=NN+Norm+LR_TKO
# BATCHSIZE=3
# LOSS_MODE="low-res"
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-$LOSS_MODE
# OUTPUT_NOISE=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX $LOSS_MODE \n\n" >> $OUTPUT_NOISE.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET  \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True >> $OUTPUT_NOISE.txt




# # ########################################## FAN 
# https://arxiv.org/pdf/2204.12451    
##### Entire Vision Backbone E2E 
# BACKBONE_TYPE=FAN_E2E
# BATCHSIZE=6
###### Entire MLP + Channel Attention 
# BACKBONE_TYPE=FAN_MLP
# BATCHSIZE=6

# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX :: $BACKBONE_TYPE $DUMP_FOLDER \n\n" >> $OUTPUT.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True >> $OUTPUT.txt



# # # ########################################## REGISTER   
# # https://arxiv.org/pdf/2309.16588
# BACKBONE_TYPE=REGISTER_E2E
# BATCHSIZE=6
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-L2 
# OUTPUT=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX :: $BACKBONE_TYPE $DUMP_FOLDER \n\n" >> $OUTPUT.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
#   MODEL.DYHEAD.NUM_CLASSES 10 MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS True >> $OUTPUT.txt








# # ########################################## MINT (Only Layer Norm)  
# ###### Mint :: 
# BACKBONE_TYPE=MINT
# BATCHSIZE=15
# TECHNIQUE=GeneralizedVLRCNN_VAR
# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER
# printf "\n\n #### $SUFFIX :: $BACKBONE_TYPE $DUMP_FOLDER \n\n" >> $OUTPUT.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True >> $OUTPUT.txt






# ########################################## RobustSAM

BATCHSIZE=4
BACKBONE_TYPE=RobustSAM
TECHNIQUE=GeneralizedVLRCNN_RobustSAM
N_BR=3

##### Doesnt influence box predictions 
# MODE="aggregate"
# SUFFIX=AG-$SUFFIX-$N_BR
# MODE="aggregate_all"
# SUFFIX=AG-ALL-$SUFFIX-$N_BR

# MODE="hidden"
# SUFFIX=HD-$SUFFIX-$N_BR 

# MODE="hidden_all"
# SUFFIX=HD-ALL-$SUFFIX-$N_BR 

# NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
# OUTPUT=ucf_output/$NAME
# DUMP_FOLDER=EXPS/$NAME
# rm -rf $DUMP_FOLDER


# printf "\n\n #### $SUFFIX :: $BACKBONE_TYPE $DUMP_FOLDER \n\n" >> $OUTPUT.txt
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --no-display \
#   --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
#   SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
#   DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST  \
#   MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
#   MODEL.RPN.NO_FUSE $MODE MODEL.RPN.FREEZE False DATASETS.REPLICA_END True DATASETS.N_BR $N_BR >> $OUTPUT.txt










rsync -a ucf_output/* ucf2:~/robustness_object_detection/GLIP/ucf_output/
# rsync -a ucf0:~/robustness_object_detection/GLIP/ucf_output/* ~/robustness_object_detection/GLIP/ucf_output/



# cd ~/robustness_object_detection/GLIP/
# sbatch Script/train_baseline.sh


