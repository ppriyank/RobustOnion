#!/bin/bash

#SBATCH --job-name=TP_6
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#SBATCH --exclude=c4-4,c5-2
#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G --cpus-per-gpu=8
#SBATCH -p gpu --qos=short 
#SBATCH -p gpu --qos=day

#################SBATCH -C gmem48 --gres=gpu:2 --mem-per-cpu=6G -c8
################SBATCH -p gpu --qos=preempt



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

# wget https://huggingface.co/genjib/AVSiam/resolve/main/jx_vit_base_patch16_224_in21k-e5005f0a.pth
# mv jx_vit_base_patch16_224_in21k-e5005f0a.pth MODEL/

BATCHSIZE=1
NUM_GPU=2
NO_OF_FRAMES=8
NO_OF_FRAMES=3

# MODEL_NAME='CAVMAE_BASE'
# OUTPUT_NOISE=CAV_M1
# N_CANDIDATES=1

MODEL_NAME='CAVMAE_BASE-RED'
OUTPUT_NOISE=CAV_M1_RED
N_CANDIDATES=2
# N_CANDIDATES=1 


FEATS_PATH=~/robustness_object_detection/Analysis/Feats/Layer
FEATS_PATH=~/robustness_object_detection/Analysis/Feats/mean_backbone
FEATS_PATH=~/robustness_object_detection/Analysis/Feats/Noise-Tra_Spa_Feat
FEATS_PATH=~/robustness_object_detection/Analysis/Feats/Spatial
EXT_CSV='../Analysis/proposed-summary-external.csv'


TARGET_CSV='../Analysis/GLIP-T_5-VPT-summary.csv'
NAME=VPT

TARGET_CSV='../Analysis/GLIP-T_5-LoRA-summary.csv'
NAME=LoRA

TARGET_CSV='../Analysis/GLIP-T_5-Adapter-summary.csv'
NAME=Adapt

# TARGET_CSV='../Analysis/GLIP-T_5-LR_TKO-summary.csv'
# NAME=Lrtk0


OUTPUT_NOISE=$OUTPUT_NOISE-$NAME-"N=$N_CANDIDATES"-"S=$NO_OF_FRAMES"
# OUTPUT_NOISE=$OUTPUT_NOISE-$NAME


################################################# Train ##################################################
#### Margin based loss 

# LOSS='margin'
# OUTPUT_NOISE=$OUTPUT_NOISE-Marg

# LOSS='clip'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS

LOSS='margin-m'
OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS

# LOSS='margin-l1'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS

# LOSS='ntext-margin'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS



###### Vanilla 1 v $N_CANDIDATES :: $NO_OF_FRAMES frames 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_feature_networks.py  \
    --override_output_dir $OUTPUT_NOISE \
    MODEL.BACKBONE $MODEL_NAME MODEL.FACTOR 4 MODEL.SAMPLE $NO_OF_FRAMES MODEL.LOSS $LOSS \
    SOLVER.TARGET_CSV $TARGET_CSV SOLVER.FEAT_PATH $FEATS_PATH DATALOADER.N_CANDIDATES $N_CANDIDATES \
    SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU >> ucf_output/$OUTPUT_NOISE.txt




# DATALOADER.NAME Pretrained 
# --vanilla_testing 


################################################# EVAL ##################################################
### Margin based loss 

# LOSS='margin'
# OUTPUT_NOISE=$OUTPUT_NOISE-Marg

# LOSS='clip'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS

# LOSS='margin-m'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS

# LOSS='margin-m'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS-Fix 
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS


# LOSS='margin-l1'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS

# LOSS='ntext-margin'
# OUTPUT_NOISE=$OUTPUT_NOISE-$LOSS


NUM_GPU=1
WT=$OUTPUT_NOISE/model_best.pth
ls $WT


####### External dataset 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_feature_networks.py  \
    --test  MODEL.WEIGHT $WT TEST.EXT_CSV $EXT_CSV \
    MODEL.BACKBONE $MODEL_NAME MODEL.FACTOR 4 MODEL.SAMPLE 8 MODEL.LOSS 'margin' \
    SOLVER.TARGET_CSV $TARGET_CSV SOLVER.FEAT_PATH $FEATS_PATH \
    TEST.IMS_PER_BATCH $NUM_GPU 

# ####### External dataset w/ Categories
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_feature_networks.py  \
    --test  MODEL.WEIGHT $WT TEST.EXT_CSV $EXT_CSV \
    MODEL.BACKBONE $MODEL_NAME MODEL.FACTOR 4 MODEL.SAMPLE 8 MODEL.LOSS 'margin' \
    SOLVER.TARGET_CSV $TARGET_CSV SOLVER.FEAT_PATH $FEATS_PATH \
    TEST.IMS_PER_BATCH $NUM_GPU DATALOADER.CATEGORIES True 

  
rsync -r $OUTPUT_NOISE ucf2:~/robustness_object_detection/GLIP/
# rsync -r ucf0:~/robustness_object_detection/GLIP/$OUTPUT_NOISE ~/robustness_object_detection/GLIP/




rsync -a ucf_output/* ucf2:~/robustness_object_detection/GLIP/ucf_output/
# rsync -a ucf0:~/robustness_object_detection/GLIP/ucf_output/* ~/robustness_object_detection/GLIP/ucf_output/



# cd ~/robustness_object_detection/GLIP/
# sbatch Script/train_toy_prob.sh



# - reduce decoder
# - pretrained blocks 
# - 3 vs 1 
# - reduce the number of frames 
# remove pretrained weights 

# train / evaluate model on kitti datasets 
# evaluate kitty test
# train on entire kitty
# Feature Dump per folder 






# - remove margin loss 