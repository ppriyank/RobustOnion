
cd ~/robustness_object_detection/MMGDINO/
conda activate OD

PORT=12345 
NUM_GPU=1
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-1} 
ENV=nccl

ROOT=/home/c3-0/datasets/
ROOT=/data/priyank/synthetic/


# ################################ "GDINO-T	Swin-T"
CONFIG=configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py
CONFIG_ODIN=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw13.py
CONFIG_ODIN35=configs/mm_grounding_dino/odinw/grounding_dino_swin-t_pretrain_odinw35.py
WT=WTS/groundingdino_swint_ogc_mmdet-822d7e9d.pth
CONFIG_LVIS=configs/mm_grounding_dino/lvis/grounding_dino_swin-t_pretrain_zeroshot_mini-lvis.py
OUTPUT=ucf_output/GDINO-T.txt
CONFIG_FLK=configs/mm_grounding_dino/flickr30k/grounding_dino_swin-t-pretrain_flickr30k.py


MODEL=GroundingDINO_MODIFIED
RUNNER=Runner_Modified
INTERVAL=4000
DATALOADER=COCODataset_Perturb


#### odinw13 - dummy run 
DATALOADER=COCODataset_Dummy
python -W ignore tools/test.py $CONFIG_ODIN $WT --root $ROOT --runner $RUNNER --no-display --sub_dataset $DATALOADER --cfg-options \
  default_hooks.logger.interval=$INTERVAL model.type=$MODEL 
  

    
################# COCO + Feature Dump 
MODEL=GroundingDINO_DUMP
RUNNER=Runner_Dump
INTERVAL=4000
DATALOADER=COCODataset_Perturb
GPU_DEVICE=0

# "GDINO-T	Swin-T"
CONFIG=configs/grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_cap4m.py
WT=WTS/groundingdino_swint_ogc_mmdet-822d7e9d.pth
OUTPUT=GDINO-T

# MM-GDINO-T ## O365,GoldG
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg.py
WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth
OUTPUT=MM-GDINO-T_O-G

# MM-GDINO-T ## O365,GoldG,GRIT,V3Det
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det.py
WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
OUTPUT=MM-GDINO-T_O-G-GR-V3

# MM-GDINO-T   ## O365,GoldG,GRIT
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m.py
WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_20231128_200818-169cc352.pth
OUTPUT=MM-GDINO-T_O-G-GR

# MM-GDINO-T ## O365,GoldG,V3Det
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py
WT=WTS/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth
OUTPUT=MM-GDINO-T_O-G-V3

# MM-GDINO-B  ## O365,GoldG,V3Det
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py
WT=WTS/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
OUTPUT=MM-GDINO-B_O-G-V3

# MM-GDINO-B* - ALL  ## O365,ALL
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_all.py
WT=WTS/grounding_dino_swin-b_pretrain_all-f9818a7c.pth
OUTPUT=MM-GDINO-B-STAR_O-ALL

# MM-GDINO-L ## O365V2,OpenImageV6,GoldG
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py
WT=WTS/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
OUTPUT=MM-GDINO-L_Ov2-OI-G

# MM-GDINO-L* - ## O365V2,OpenImageV6,ALL
CONFIG=configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py
WT=WTS/grounding_dino_swin-l_pretrain_all-56d69e78.pth
OUTPUT=MM-GDINO-L-STAR_Ov2-OI-ALL

NOISES=("low-res" "motion_blur2")
for MODE in "${NOISES[@]}"
do
  ###### Spatial Tokens (Dumps only 0th layer)
  CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tools/test.py $CONFIG $WT --runner $RUNNER --no-display --root $ROOT \
    --work-dir logs/$OUTPUT/$MODE --cfg-options test_dataloader.dataset.mode=$MODE \
    model.type=$MODEL test_dataloader.dataset.type=$DATALOADER default_hooks.logger.interval=$INTERVAL \
    test_cfg.dump_index=0, test_cfg.backbone_only=True model.backbone.type=SwinTransformer_Dump
done

MODE="atmospheric"
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tools/test.py $CONFIG $WT --runner $RUNNER --no-display --root $ROOT \
  --work-dir logs/$OUTPUT/$MODE --cfg-options test_dataloader.dataset.mode=$MODE env_cfg.mp_cfg.mp_start_method="spawn" env_cfg.mp_cfg.opencv_num_threads=4 \
  model.type=$MODEL test_dataloader.dataset.type=$DATALOADER default_hooks.logger.interval=$INTERVAL \
  test_cfg.dump_index=0, test_cfg.backbone_only=True model.backbone.type=SwinTransformer_Dump
        
# Vanilla 
MODE=NONE
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python tools/test.py $CONFIG $WT --runner $RUNNER --no-display --root $ROOT \
  --work-dir logs/$OUTPUT/$MODE --cfg-options \
  model.type=$MODEL test_dataloader.dataset.type=$DATALOADER default_hooks.logger.interval=$INTERVAL \
  test_cfg.dump_index=0, test_cfg.backbone_only=True model.backbone.type=SwinTransformer_Dump
        

