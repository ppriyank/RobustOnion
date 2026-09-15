#!/bin/bash

#SBATCH --job-name=GL_2
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu

#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G --cpus-per-gpu=8
####SBATCH -C gmem80 --gres=gpu:1 --mem-per-cpu=6G -c8


################SBATCH -p preempt --qos=preempt
#################SBATCH -p gpu --qos=day
################SBATCH -p gpu --qos=short 
#################SBATCH -C gmemT48 --gres=gpu:turing:1 --mem-per-cpu=6G -c8

# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --qos preempt bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash 
# srun --pty --gres=gpu:2 --cpus-per-gpu=8 -C gmem48 bash 
# srun --pty --gres=gpu:turing:2 --cpus-per-gpu=8 -C gmemT48 --qos preempt bash 
# srun --pty --gres=gpu:turing:2 --cpus-per-gpu=8 -C gmemT48 bash 

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
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/robustness_object_detection/GLEE/ucf_output/

module load cuda/12.1
conda activate GLEE
cd ~/robustness_object_detection/GLEE/


PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"

export DETECTRON2_DATASETS=/home/c3-0/datasets/
NUM_GPU=1



###########################################################################################################
################################### LITE 
###########################################################################################################

##################### GLLE_LITE_JOINT (STAGE 1) ########################################################################################
CONFIG=projects/GLEE/configs/images/Lite/Stage1_pretrain_openimage_obj365_CLIPfrozen_R50.yaml 
WT=Weights/GLEE_Lite_pretrain.pth
OUTPUT=ucf_output/GLLE_LITE_STAGE1.txt

##################### GLLE_LITE_JOINT (STAGE 2) ########################################################################################
CONFIG=projects/GLEE/configs/images/Lite/Stage2_joint_training_CLIPteacher_R50.yaml
WT=Weights/GLEE_Lite_joint.pth
OUTPUT=ucf_output/GLLE_LITE_STAGE2.txt

##################### GLLE_LITE_JOINT (STAGE 3) ########################################################################################
CONFIG=projects/GLEE/configs/images/Lite/Stage3_scaleup_CLIPteacher_R50.yaml 
WT=Weights/GLEE_Lite_scaleup.pth
OUTPUT=ucf_output/GLLE_LITE_STAGE3.txt


# ############################################################################################################
# #################################### PLUS 
# ############################################################################################################

###################### GLLE_PLUS_JOINT (STAGE 1) ########################################################################################
CONFIG=projects/GLEE/configs/images/Plus/Stage1_pretrain_openimage_obj365_CLIPfrozen_SwinL.yaml 
WT=Weights/GLEE_Plus_pretrain.pth
OUTPUT=ucf_output/GLLE_PLUS_STAGE1.txt

###################### GLLE_PLUS_JOINT (STAGE 2) ########################################################################################
CONFIG=projects/GLEE/configs/images/Plus/Stage2_joint_training_CLIPteacher_SwinL.yaml
WT=Weights/GLEE_Plus_joint.pth
OUTPUT=ucf_output/GLLE_PLUS_STAGE2.txt

###################### GLLE_PLUS_JOINT (STAGE 3) ########################################################################################
CONFIG=projects/GLEE/configs/images/Plus/Stage3_scaleup_CLIPteacher_SwinL.yaml
WT=Weights/GLEE_Plus_scaleup.pth
OUTPUT=ucf_output/GLLE_PLUS_STAGE3.txt



############################################################################################################
#################################### PRO 
############################################################################################################

###################### GLLE_PRO_JOINT (STAGE 1) ########################################################################################
CONFIG=projects/GLEE/configs/images/Pro/Stage1_pretrain_openimage_obj365_CLIPfrozen_EVA02L_LSJ1536.yaml 
WT=Weights/GLEE_Pro_pretrain.pth
OUTPUT=ucf_output/GLLE_PRO_STAGE1.txt

##################### GLLE_PRO_JOINT (STAGE 2) ########################################################################################
CONFIG=projects/GLEE/configs/images/Pro/Stage2_joint_training_CLIPteacher_EVA02L.yaml
WT=Weights/GLEE_Pro_joint.pth
OUTPUT=ucf_output/GLLE_PRO_STAGE2.txt

#################### GLLE_PRO_JOINT (STAGE 3) ########################################################################################
CONFIG=projects/GLEE/configs/images/Pro/Stage3_scaleup_CLIPteacher_EVA02L.yaml
WT=Weights/GLEE_Pro_scaleup.pth
OUTPUT=ucf_output/GLLE_PRO_STAGE3.txt



# # ###### COCO 
# printf "\n\n #### OG \n\n" >> $OUTPUT
# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  DATASETS.TEST '("coco_2017_val",)' >> $OUTPUT


# # ###### COCO DUMP FETURES +  Low Res + Motion Blur 
# DATALOADER=RefCOCODatasetMapper_Perturb
# MODE="low-res"
# SEVs=(0 1 2 3 4 5)
# MODE="motion_blur2"
# SEVs=(3)
# NUM_GPU=1
# DUMP=VANILLA-$MODE
# for SEV in "${SEVs[@]}"
#     do
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats \
#         --dump-start 0 --dump-end 1000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 
    
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats \
#         --dump-start 1000 --dump-end 2000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats \
#         --dump-start 2000 --dump-end 3000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats \
#         --dump-start 3000 --dump-end 4000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats \
#         --dump-start 4000 --dump-end 5000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 
# done
# # rsync -a ucf0:/home/ppriyank/robustness_object_detection/GLEE/VANILLA* ~/robustness_object_detection/GLEE/ucf_output/

# # ###### COCO DUMP FETURES +  Atmospheric
# DATALOADER=RefCOCODatasetMapper_Perturb
# MODE="atmospheric"    
# NUM_GPU=1
# DUMP=VANILLA-$MODE
# SEV=3
# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
#     --dump-start 0 --dump-end 1000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
#     --dump-start 1000 --dump-end 2000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
#     --dump-start 2000 --dump-end 3000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
#     --dump-start 3000 --dump-end 4000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 

# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --dump-feats --spawn-method \
#     --dump-start 4000 --dump-end 5000 OUTPUT_DIR $DUMP MODEL.WEIGHTS $WT DATASETS.TEST '("coco_2017_val",)' DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE SEV $SEV MODEL.DUMP_FEATS True MODEL.META_ARCHITECTURE GLEE_DUMP MODEL.BACKBONE.NAME D2_EVA02_DUMP 


    
     
  
    



##### Perturbations
# DATALOADER=RefCOCODatasetMapper_Perturb



# ###################### Noise  - 1
# NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  \
#     DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATASETS.TEST '("coco_2017_val",)' >> $OUTPUT
# done



# ###################### Noise  - 2
# NOISES=("rain_model2" "snow_model2" "atmospheric")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT
#   ####### Vanilla Training 
#   python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --spawn-method --eval-only  MODEL.WEIGHTS $WT  \
#     DATASETS.TEST_DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE DATASETS.TEST '("coco_2017_val",)'  >> $OUTPUT
#     # DATALOADER.NUM_WORKERS 0 \
#     # >> $OUTPUT
# done




##### LVIS  
# OUTPUT=$OUTPUT-LVIS-2 
# printf "\n\n #### OG LVIS \n\n" >> $OUTPUT
# printf "\n\n MINi LVIS \n\n" >> $OUTPUT
# python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  DATASETS.TEST '("lvis_v1_minival",)' >> $OUTPUT

###### LVIS + NOISE 
# OUTPUT=$OUTPUT-PERTURB-LVIS
# MODE="low-res"
# NUM_GPU=1
# SEVs=(4 5)
# # SEVs=(0 1 2 3 4 5)
# for SEV in "${SEVs[@]}"
# do
#     printf "\n\n #### MINi LVIS + $MODE SEV :: $SEV  \n\n" >> $OUTPUT
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  \
#         DATASETS.TEST '("lvis_v1_minival",)' DATASETS.DATALOADER_MODE $MODE SEV $SEV >> $OUTPUT
# done 

# OUTPUT=$OUTPUT-NOISE-LVIS
# NUM_GPU=1

# NOISES=("NONE" "motion_blur2")
# for MODE in "${NOISES[@]}"
# do
#     printf "\n\n #### MINi LVIS + $MODE \n\n" >> $OUTPUT
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  \
#         DATASETS.TEST '("lvis_v1_minival",)' DATASETS.DATALOADER_MODE $MODE >> $OUTPUT
# done 
# NOISES=("atmospheric")
# for MODE in "${NOISES[@]}"
# do
#     printf "\n\n #### MINi LVIS + $MODE \n\n" >> $OUTPUT
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --spawn-method --eval-only  MODEL.WEIGHTS $WT  \
#         DATASETS.TEST '("lvis_v1_minival",)' DATASETS.DATALOADER_MODE $MODE >> $OUTPUT
# done 

  

# # ###### ODINW-13  
# OUTPUT=$OUTPUT-ODINW13 
# TEST_DATASET='("odinw13_AerialDrone","odinw13_Aquarium","odinw13_Rabbits","odinw13_EgoHands","odinw13_Mushrooms","odinw13_Packages","odinw13_PascalVOC","odinw13_Pistols","odinw13_Pothole","odinw13_Raccoon","odinw13_Shellfish","odinw13_Thermal","odinw13_Vehicles",)' 
# printf "\n\n #### OG ODINW-13 \n\n" >> $OUTPUT
# python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET >> $OUTPUT


# # # ###### ODINW-13 + Noise 
# NUM_GPU=1
# OUTPUT=$OUTPUT-NOISE-ODINW13 
# TEST_DATASET='("odinw13_AerialDrone","odinw13_Aquarium","odinw13_Rabbits","odinw13_EgoHands","odinw13_Mushrooms","odinw13_Packages","odinw13_PascalVOC","odinw13_Pistols","odinw13_Pothole","odinw13_Raccoon","odinw13_Shellfish","odinw13_Thermal","odinw13_Vehicles",)' 

# NOISES=("NONE" "motion_blur2")
# for MODE in "${NOISES[@]}"
# do
#     printf "\n\n\n\n #### OG ODINW-13 $MODE \n\n\n\n" >> $OUTPUT
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  \
#         DATASETS.TEST $TEST_DATASET DATASETS.DATALOADER_MODE $MODE  >> $OUTPUT
#     python Script/avg_odin.py $OUTPUT 13 >> $OUTPUT
# done 

# NOISES=("atmospheric")
# for MODE in "${NOISES[@]}"
# do
#     printf "\n\n\n\n #### OG ODINW-13 $MODE \n\n\n\n" >> $OUTPUT
#     python projects/GLEE/train_net.py --config-file $CONFIG  --num-gpus $NUM_GPU --no-display --spawn-method --eval-only  MODEL.WEIGHTS $WT  \
#         DATASETS.TEST $TEST_DATASET DATASETS.DATALOADER_MODE $MODE  >> $OUTPUT
#     python Script/avg_odin.py $OUTPUT 13 >> $OUTPUT
# done 


# # ###### FLICKER
# OUTPUT=$OUTPUT-FLICKER
# TEST_DATASET='("flicker-test",)' 
# printf "\n\n #### FLICKER \n\n" >> $OUTPUT
# python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only  MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET >> $OUTPUT




# ###### EXTERNAL 
OUTPUT_EXT=$OUTPUT-EXT
MAPPER='ext_dataset'

### BDDK-100 
TEST_DATASET='("bddk-test",)' 
printf "\n\n #### BDDK \n\n" >> $OUTPUT_EXT
python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only \
  MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET INPUT.DATASET_MAPPER_NAME $MAPPER >> $OUTPUT_EXT
 
 
# python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --eval-only \
#   MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET INPUT.DATASET_MAPPER_NAME $MAPPER >> $OUTPUT_EXT

# TEST_DATASET='("dawn-test",)' 
# printf "\n\n #### DAWN \n\n" >> $OUTPUT_EXT
# python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only \
#   MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET INPUT.DATASET_MAPPER_NAME $MAPPER >> $OUTPUT_EXT


# TEST_DATASET='("wedge-test",)' 
# printf "\n\n #### WEDGE \n\n" >> $OUTPUT_EXT
# python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only \
#   MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET INPUT.DATASET_MAPPER_NAME $MAPPER >> $OUTPUT_EXT




###################### FoggyCitiscape
# _foggy_beta_0.005.png #### LESS FOG 
# _foggy_beta_0.01.png #### MEDIUM FOG 
# _foggy_beta_0.02.png ##### DENSE FOG 

LEVEL=(0.005 0.01 0.02)
for FOG in "${LEVEL[@]}"
do
  # # FoggyCitiscape amodal
  TEST_DATASET='("FoggyCitiscape-a-modal-test",)' 
  printf "\n\n #### FoggyCitiscape (Amodal) FOG - Level : $FOG \n\n" >> $OUTPUT_EXT  
  python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only \
    MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET INPUT.DATASET_MAPPER_NAME $MAPPER TEST.FOG_LEVEL $FOG >> $OUTPUT_EXT


  TEST_DATASET='("FoggyCitiscape-modal-test",)' 
  printf "\n\n #### FoggyCitiscape (modal) FOG - Level : $FOG  \n\n" >> $OUTPUT_EXT
  python projects/GLEE/train_net.py --config-file $CONFIG --num-gpus $NUM_GPU --no-display --eval-only \
    MODEL.WEIGHTS $WT  DATASETS.TEST $TEST_DATASET INPUT.DATASET_MAPPER_NAME $MAPPER TEST.FOG_LEVEL $FOG >> $OUTPUT_EXT
done















    
    
    

rsync -a ucf_output/* ucf2:~/robustness_object_detection/GLEE/ucf_output/
# rsync -a ucf0:~/robustness_object_detection/GLEE/ucf_output/* ~/robustness_object_detection/GLEE/ucf_output/



# cd ~/robustness_object_detection/GLEE/
# sbatch Scripts/vanilla_benchmark.sh
