#!/bin/bash

#SBATCH --job-name=OR_MG_2
#SBATCH --output=ucf_output/slurm-%j.out
#SBATCH --gres-flags=enforce-binding
#SBATCH -p gpu
#################SBATCH --exclude=c4-4
#SBATCH -C gmem48 --gres=gpu:1 --mem-per-cpu=6G --cpus-per-gpu=8

################SBATCH -C gmem32 --gres=gpu:1 -c8
###############SBATCH -p gpu --qos=day
##############SBATCH -p gpu --qos=short 

# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 --qos preempt bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash 
# srun --pty --gres=gpu:1 --cpus-per-gpu=8 -C gmem48 bash --qos preempt bash 
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
rsync -a ucf_output/slurm-$SLURM_JOB_ID.sh ucf2:~/robustness_object_detection/MMGDINO/ucf_output/

module load cuda/11.3
conda activate OD
cd ~/robustness_object_detection/MMGDINO/

PORT=$((RANDOM % 55 + 12345))
while ss -tuln | grep -q ":$PORT"; do
  PORT=$((RANDOM % 55 + 12345))
done
echo "Free port found: $PORT"


NUM_GPU=1
ROOT=/home/c3-0/datasets/


# ################################ "GDINO-T	Swin-T"
CONFIG=configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py
CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py
CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
WT=WTS/groundingdino_swint_ogc_mmdet-822d7e9d.pth
CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
OUTPUT=ucf_output/GDINO-T.txt
CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py

# # ################################### MM-GDINO-T ## O365,GoldG,GRIT,V3Det
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py
# WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
# OUTPUT=ucf_output/MM-GDINO-T_O-G-GR-V3.txt
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py

# # ################################# MM-GDINO-T   ## O365,GoldG,GRIT
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m.py
# WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
# OUTPUT=ucf_output/MM-GDINO-T_O-G-GR.txt
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py

# # ################################## MM-GDINO-T ## O365,GoldG,V3Det
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py
# WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
# OUTPUT=ucf_output/MM-GDINO-T_O-G-V3.txt
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py

# # ################################ MM-GDINO-T ## O365,GoldG
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg.py
# WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
# OUTPUT=ucf_output/MM-GDINO-T_O-G.txt
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py






# # # #################################### MM-GDINO-B  ## O365,GoldG,V3Det
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py
# WT=WTS/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-b_all-finetune_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-b_all-finetune_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-b_pretrain_zeroshot_mini-lvis.py
# OUTPUT=ucf_output/MM-GDINO-B_O-G-V3.txt
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-b-pretrain_flickr30k.py

# # ############### MM-GDINO-B* - ALL  ## O365,ALL
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_all.py
# WT=WTS/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-b_all-finetune_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-b_all-finetune_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-b_pretrain_zeroshot_mini-lvis.py
# OUTPUT=ucf_output/MM-GDINO-B-STAR_O-ALL.txt
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-b-pretrain_flickr30k.py


# # ##################################### MM-GDINO-L ## O365V2,OpenImageV6,GoldG
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py
# WT=WTS/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
# OUTPUT=ucf_output/MM-GDINO-L_Ov2-OI-G.txt
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-l_all-finetune_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-l_all-finetune_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-l_pretrain_zeroshot_mini-lvis.py
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-l-pretrain_flickr30k.py


# # # ###################################### MM-GDINO-L* - ## O365V2,OpenImageV6,ALL
# CONFIG=configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py
# WT=WTS/grounding_dino_swin-l_pretrain_all-56d69e78.pth
# OUTPUT=ucf_output/MM-GDINO-L-STAR_Ov2-OI-ALL.txt
# CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-l_all-finetune_odinw13.py
# CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-l_all-finetune_odinw35.py
# CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-l_pretrain_zeroshot_mini-lvis.py
# CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-l-pretrain_flickr30k.py


MODEL=GroundingDINO_MODIFIED
RUNNER=Runner_Modified
INTERVAL=4000
DATALOADER=COCODataset_Perturb


# # COCO 
# printf "\n\n #### OG \n\n" >> $OUTPUT   
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER \
#         --cfg-options default_hooks.logger.interval=$INTERVAL model.type=$MODEL >> $OUTPUT


################################################################# 
##### Perturbations
################################################################# 

# ###### Trial NO Transform 
# MODE="None"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT


# ################# NOISES
# ###### Pixel Dropout
# MODE="pixel_dropout"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT


# ###### ISO Blur 
# MODE="iso_blur"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT

# ###### Salt-pepper 
# MODE="salt_pepper"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT


################# REAL WORLD 
###### Low Res 
# MODE="low-res"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT

# ###### Focus Blur 
# MODE="focus_blur"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT

# ###### Gaussian Blur 
# MODE="gaussian_blur"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT

# ###### JPG Compression 
# MODE="jpg_compression"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT

# ###### Chromatic  
# MODE="chromatic"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT

# ###### Motion Blur 
# MODE="motion_blur2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT


################# WEATHER 

# ###### Fog 
# MODE="fog"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL >> $OUTPUT


# ###### Rain  
# MODE="rain_model2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL \
#   env_cfg.mp_cfg.mp_start_method="spawn" env_cfg.mp_cfg.opencv_num_threads=4 >> $OUTPUT

# ###### SNOW 
# MODE="snow_model2"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL \
#   env_cfg.mp_cfg.mp_start_method="spawn" env_cfg.mp_cfg.opencv_num_threads=4 >> $OUTPUT

# ###### Atmospheric 
# MODE="atmospheric"
# printf "\n\n #### $MODE \n\n" >> $OUTPUT
# python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL test_dataloader.dataset.type=$DATALOADER \
#   test_dataloader.dataset.mode=$MODE model.type=$MODEL \
#   env_cfg.mp_cfg.mp_start_method="spawn" env_cfg.mp_cfg.opencv_num_threads=4 >> $OUTPUT


# #### odinw13
# INTERVAL=4000
# OUTPUT_ODIN=$OUTPUT-ODWIN13
# python -W ignore tools/test.py $CONFIG_ODIN $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL model.type=$MODEL >> $OUTPUT_ODIN
# python Script/avg_odin.py $OUTPUT_ODIN 13 >> $OUTPUT_ODIN


# ##### odinw13 + Noise 
# DATALOADER=COCODataset_Perturb
# INTERVAL=4000
# OUTPUT_ODIN=$OUTPUT-NOISE-ODWIN13
# NOISES=("NONE" "motion_blur2")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n\n\n #### $MODE \n\n\n\n" >> $OUTPUT_ODIN
#   python -W ignore tools/test.py $CONFIG_ODIN $WT --root $ROOT --runner $RUNNER --no-display --sub_dataset $DATALOADER --cfg-options \
#     default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.mode=$MODE >> $OUTPUT_ODIN
#     python Script/avg_odin.py $OUTPUT_ODIN 13 >> $OUTPUT_ODIN
# done

# OUTPUT_ODIN=$OUTPUT-NOISE-LR2
# MODE="low-res"
# SEVS=(0 1 2 3 4 5)
# for SEV in "${SEVS[@]}"
# do
#   printf "\n #### $SEV \n" >> $OUTPUT_ODIN
#   printf "\n\n\n\n #### $MODE $SEV \n\n\n\n" >> $OUTPUT_ODIN
#   python -W ignore tools/test.py $CONFIG_ODIN $WT --root $ROOT --runner $RUNNER --no-display --sub_dataset $DATALOADER --severity $SEV --cfg-options \
#     default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.mode=$MODE >> $OUTPUT_ODIN
#   python Script/avg_odin.py $OUTPUT_ODIN 13 >> $OUTPUT_ODIN
# done


# NOISES=("atmospheric")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n\n\n #### $MODE \n\n\n\n" >> $OUTPUT_ODIN
#   python -W ignore tools/test.py $CONFIG_ODIN $WT --root $ROOT --runner $RUNNER --no-display --sub_dataset $DATALOADER --cfg-options \
#     default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.mode=$MODE \
#     env_cfg.mp_cfg.mp_start_method="spawn" env_cfg.mp_cfg.opencv_num_threads=4 >> $OUTPUT_ODIN
#     # python Script/avg_odin.py $OUTPUT_ODIN 13 >> $OUTPUT_ODIN
# done




# # #### odinw35
# INTERVAL=4000
# OUTPUT_ODIN=$OUTPUT-ODWIN35
# python -W ignore tools/test.py $CONFIG_ODIN35 $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL model.type=$MODEL >> $OUTPUT_ODIN
# python Script/avg_odin.py $OUTPUT_ODIN 35 >> $OUTPUT_ODIN




# # ##### LVIS (MINI LVIS)
# INTERVAL=4000
# OUTPUT_LVIS=$OUTPUT-LVIS
# python -W ignore tools/test.py $CONFIG_LVIS $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL model.type=$MODEL >> $OUTPUT_LVIS


# # ##### LVIS (MINI LVIS) + Noise 
# DATALOADER=LVISDataset_Perturb
# INTERVAL=4000

# OUTPUT_LVIS=$OUTPUT-Noise-LVIS
# NOISES=("NONE" "motion_blur2")
# # NOISES=("motion_blur2")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT_LVIS
#   python -W ignore tools/test.py $CONFIG_LVIS $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#     default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.mode=$MODE test_dataloader.dataset.type=$DATALOADER >> $OUTPUT_LVIS
# done
# NOISES=("atmospheric")
# for MODE in "${NOISES[@]}"
# do
#   printf "\n\n #### $MODE \n\n" >> $OUTPUT_LVIS
#   python -W ignore tools/test.py $CONFIG_LVIS $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#     default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.mode=$MODE test_dataloader.dataset.type=$DATALOADER \
#     env_cfg.mp_cfg.mp_start_method="spawn" env_cfg.mp_cfg.opencv_num_threads=4 >> $OUTPUT_LVIS
# done


# # ##### FLICKER 
# INTERVAL=4000
# OUTPUT_FLK=$OUTPUT-FLICKR
# python -W ignore tools/test.py $CONFIG_FLK $WT --root $ROOT --runner $RUNNER --no-display --cfg-options \
#   default_hooks.logger.interval=$INTERVAL model.type=$MODEL >> $OUTPUT_FLK



####### External Dataset 
ROOT=/data/priyank/synthetic/
ROOT=/home/c3-0/datasets/
DATALOADER=Extern_Dataset
OUTPUT_EXT=$OUTPUT-EXTERNAL


# # BDDK-100
printf "\n\n #### BDDK \n\n" >> $OUTPUT_EXT   
CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER \
    --diff-dataset BDDK --cfg-options default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
    test_dataloader.dataset.type=$DATALOADER >> $OUTPUT_EXT

# # DAWN
# printf "\n\n #### DAWN \n\n" >> $OUTPUT_EXT
# CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER \
#     --diff-dataset DAWN --cfg-options default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.type=$DATALOADER >> $OUTPUT_EXT
 
# # WEDGE
# printf "\n\n #### WEDGE \n\n" >> $OUTPUT_EXT
# CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER \
#     --diff-dataset WEDGE --cfg-options default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
#     test_dataloader.dataset.type=$DATALOADER >> $OUTPUT_EXT
  


###################### FoggyCitiscape
# _foggy_beta_0.005.png #### LESS FOG 
# _foggy_beta_0.01.png #### MEDIUM FOG 
# _foggy_beta_0.02.png ##### DENSE FOG 

LEVEL=(0.005 0.01 0.02)
for FOG in "${LEVEL[@]}"
do
  # # FoggyCitiscape amodal
  printf "\n\n #### FoggyCitiscape (Amodal) FOG - Level : $FOG \n\n" >> $OUTPUT_EXT
  CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER \
    --diff-dataset FoggyCity_A --cfg-options default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
    test_dataloader.dataset.type=$DATALOADER test_dataloader.dataset.fog_level=$FOG >> $OUTPUT_EXT

  # # FoggyCitiscape modal  
  printf "\n\n #### FoggyCitiscape (modal) FOG - Level : $FOG  \n\n" >> $OUTPUT_EXT
  CUDA_VISIBLE_DEVICES=0 python -W ignore tools/test.py $CONFIG $WT --root $ROOT --runner $RUNNER \
    --diff-dataset FoggyCity --cfg-options default_hooks.logger.interval=$INTERVAL model.type=$MODEL \
    test_dataloader.dataset.type=$DATALOADER test_dataloader.dataset.fog_level=$FOG >> $OUTPUT_EXT  
done





  












        



rsync -a ucf_output/* ucf2:~/robustness_object_detection/MMGDINO/ucf_output/
# rsync -a ucf0:~/robustness_object_detection/MMGDINO/ucf_output/* ~/robustness_object_detection/MMGDINO/ucf_output/



# cd ~/robustness_object_detection/MMGDINO/
# sbatch Script/vanilla_benchmark.sh