#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=O3
#SBATCH --output=ucf_output_newton/slurm-%j.out
#SBATCH --constraint=gpu80
#SBATCH --exclude=evc42,evc25

###### SBATCH --partition=preemptable --qos preemptable
##### SBATCH -C '!gmem16'
###### srun --pty -t24:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu16 bash
###### srun --pty -t48:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu32 bash
###### srun --pty -t24:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu32 bash
###### srun --pty -t180:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu32 bash
###### srun --pty -t200:00:00 --gres=gpu:2 --cpus-per-gpu=6 --constraint=gpu80 bash
###### srun --pty -t1:00:00 --gres=gpu:1 --cpus-per-gpu=6 --constraint=gpu80 --exclude=evc42,evc25 bash
# srun --pty -t1:00:00 --cpus-per-task=6 bash

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






###################### GLIP-T [5] ########################################################################################

# wget https://huggingface.co/genjib/AVSiam/resolve/main/jx_vit_base_patch16_224_in21k-e5005f0a.pth
# mv jx_vit_base_patch16_224_in21k-e5005f0a.pth MODEL/

BATCHSIZE=1
NUM_GPU=2
DROPOUT=True 

DROPOUT=False 
NO_OF_FRAMES=8
NO_OF_FRAMES=3
# NO_OF_FRAMES=20

MODEL_NAME='CAVMAE_BASE'
OUTPUT_NOISE=CAV_M1
N_CANDIDATES=4

MODEL_NAME='CAVMAE_BASE-RED'
OUTPUT_NOISE=CAV_M1_RED
N_CANDIDATES=2
N_CANDIDATES=1 

FEATS_PATH=~/robustness_object_detection/Analysis/Feats/Layer
FEATS_PATH=~/robustness_object_detection/Analysis/Feats/mean_backbone
FEATS_PATH=~/robustness_object_detection/Analysis/Feats/Noise-Tra_Spa_Feat
FEATS_PATH=~/robustness_object_detection/Analysis/Feats/Spatial
EXT_CSV='../Analysis/proposed-summary-external.csv'

# CAV_M1-Lrtk0-N=4-S=3-margin-m
# CAV_M1_RED-Lrtk0-N=2-S=20-margin-m
# CAV_M1_RED-Lrtk0-N=2-S=3-margin-m
# CAV_M1_RED-VPT-N=2-S=3-margin-m
# CAV_M1_RED-Adapt-N=2-S=3-margin-m


TARGET_CSV='../Analysis/GLIP-T_5-VPT-summary.csv'
NAME=VPT

# TARGET_CSV='../Analysis/GLIP-T_5-LoRA-summary.csv'
# NAME=LoRA

TARGET_CSV='../Analysis/GLIP-T_5-Adapter-summary.csv'
NAME=Adapt

TARGET_CSV='../Analysis/GLIP-T_5-LR_TKO-summary.csv'
NAME=Lrtk0


OUTPUT_NOISE=$OUTPUT_NOISE-$NAME-"N=$N_CANDIDATES"-"S=$NO_OF_FRAMES"



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
    MODEL.BACKBONE $MODEL_NAME MODEL.FACTOR 4 MODEL.SAMPLE $NO_OF_FRAMES MODEL.LOSS $LOSS SOLVER.TARGET_CSV $TARGET_CSV SOLVER.FEAT_PATH $FEATS_PATH DATALOADER.N_CANDIDATES $N_CANDIDATES \
    SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU \
    MODEL.PRETRAIN False MODEL.DROPOUT $DROPOUT >> ucf_output_newton/$OUTPUT_NOISE.txt




# DATALOADER.NAME Pretrained 
# --vanilla_testing 
# --debug 


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
# rsync -r ucf4:~/robustness_object_detection/GLIP/$OUTPUT_NOISE ~/robustness_object_detection/GLIP/



rsync -a ucf_output_newton/* ucf2:~/robustness_object_detection/GLIP/ucf_output_newton/
# rsync -a ucf4:~/robustness_object_detection/GLIP/ucf_output_newton/* ~/robustness_object_detection/GLIP/ucf_output_newton/



# cd ~/robustness_object_detection/GLIP/
# sbatch Script/train_toy_prob_newton.sh




# - reduce the number of frames 
# remove pretrained weights 
# - remove margin loss 