# srun --pty --gres=gpu:2 --cpus-per-gpu=8 -C gmem48 bash 
conda activate OD

cd ~/robustness_object_detection/GLIP/
ENV='nccl'
NUM_GPU=1
PORT=12345
mkdir ucf_output

export DETECTRON2_DATASETS=/home/c3-0/datasets/flickr_dataset_30k/
export DATASET=/home/c3-0/datasets/flickr_dataset_30k/
export ADD_DATASET=/home/c3-0/datasets/
export TOKENIZERS_PARALLELISM=true 


export DETECTRON2_DATASETS=/data/priyank/synthetic/flickr_dataset_30k/
export DATASET=/data/priyank/synthetic/flickr_dataset_30k/
export ADD_DATASET=/data/priyank/synthetic/
export TOKENIZERS_PARALLELISM=true 




###################### GLIP-T [5] ########################################################################################
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output/GLIP-T_5.txt

CONFIG=configs/pretrain/glip_A_Swin_T_O365.yaml
WT=MODEL/glip_a_tiny_o365.pth

CONFIG=configs/pretrain/glip_Swin_T_O365.yaml
WT=MODEL/glip_tiny_model_o365.pth

CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT=MODEL/glip_tiny_model_o365_goldg.pth

CONFIG=configs/pretrain/glip_Swin_L.yaml
WT=MODEL/glip_large_model.pth







# ######### Multi-Scale Training + VPT (DEEP) + Multi-Scale Training
BACKBONE_TYPE=LR_TKO
BACKBONE_TYPE=VPT_DEEP
BACKBONE_TYPE=ADAPTER
BACKBONE_TYPE=LORA



SUBSET=subset3
DUMP_FOLDER=EXPS/BASELINE5/
rm -rf $DUMP_FOLDER
TRAIN_FN=multi_scale_train
DATALOADER=FlickrDataset_MULTI_SCALE
TEST_DATALOADER=COCODataset_Perturb
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
BATCHSIZE=3


MODE="snow_model2"
MODE="atmospheric"
MODE="low-res"

####### Vanilla model tasting (switch off and switch on)
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
    --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN --subset $SUBSET --vanilla-testing \
    SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
    DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE
# BATCHSIZE=5    
# 2025-03-19 15:27:08,053 maskrcnn_benchmark.inference INFO: OrderedDict([('bbox', OrderedDict([('AP', 0.261058442414401), ('AP50', 0.3838019012433354), ('AP75', 0.27690802505392115), ('APs', 0.09355668822144295), ('APm', 0.28398257481890615), ('APl', 0.4194956940285372)]))])
# ===== OrderedDict([('bbox', OrderedDict([('AP', 0.261058442414401), ('AP50', 0.3838019012433354), ('AP75', 0.27690802505392115), ('APs', 0.09355668822144295), ('APm', 0.28398257481890615), ('APl', 0.4194956940285372)]))])    

####### Display Image 
# --display_images "all"



####### Re-evalaute Predictions DAWN (saved tensors)
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST '("dawn_test",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False









###### COCO -- GFLOP  (VANILLA)
BATCHSIZE=2
LOSS_MODE="low-res"
SUBSET=subset3
TRAIN_FN=multi_scale_train_n_captions
DATALOADER=FlickrDataset_MULTI_SCALE_LRCAPT
TEST_DATALOADER=COCODataset_Perturb
TECHNIQUE=GeneralizedVLRCNN_TRILANG_GFLOP
HEAD=VLDyHeadModule_LANG_TRI


CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG --GFLOP  \
    --subset $SUBSET SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
    DATASETS.TRAIN_DATALOADER $DATALOADER DATASETS.TEST_DATALOADER $TEST_DATALOADER DATASETS.DATALOADER_MODE $LOSS_MODE \
    MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE MODEL.RPN_ARCHITECTURE $HEAD 
    




###### BBDK100 + Dump 
TECHNIQUE=GeneralizedVLRCNN_TRILANG_DUMP 
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST '("bdd100k_val",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  TEST.PREDICT True MODEL.META_ARCHITECTURE $TECHNIQUE 
  



###################### Noise + SPARSE DUMP 
TECHNIQUE=GeneralizedVLRCNN_TRILANG_DUMP 
DATALOADER=COCODataset_Perturb
NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
BACKBONE=SWINT-FPN-RETINANET-SPARSE
BACKBONE=SWINT-FPN-RETINANET-SPARSE2

for MODE in "${NOISES[@]}"
do
  CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "Feat_DUMP" MODEL.BACKBONE.CONV_BODY $BACKBONE
done

NOISES=("snow_model2" "atmospheric" "rain_model2")
for MODE in "${NOISES[@]}"
do
  CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP --spawn-method \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "Feat_DUMP" MODEL.BACKBONE.CONV_BODY $BACKBONE
done




# Patch Super resolution 
# # # ######### BaseLine Training (no freezing) + Multi-Scale Training + LR TOKENs  (Super resolution Patches)
# DUMP_FOLDER=BASELINE6_2/
# rm -rf $DUMP_FOLDER
# TRAIN_FN=multi_scale_train
# DATALOADER=LR_FlickrDataset_MULTI_SCALE
# TECHNIQUE=GeneralizedVLRCNN_VPT
# BATCHSIZE=2
# CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py \
#     --override_output_dir $DUMP_FOLDER --use-tensorboard --task_config $TASK_CONFIG --all-eval \
#     --severity 3 --subset=$SUBSET --train_fn=$TRAIN_FN  \
#     --config-file $CONFIG SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU \
#     TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection \
#     SOLVER.CHECKPOINT_PERIOD -1  DATASETS.DATALOADER $DATALOADER \
#     MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.LR_TKO True MODEL.BACKBONE_TYPE build_lr_tko_patch_fix












###################### Noise + GRAM MAtrix DUMP 
TECHNIQUE=GeneralizedVLRCNN_TRILANG_DUMP 
DATALOADER=COCODataset_Perturb
NOISES=("pixel_dropout" "iso_blur" "salt_pepper" "low-res" "focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
NOISES=("focus_blur" "jpg_compression" "chromatic" "motion_blur2" "fog")
for MODE in "${NOISES[@]}"
do
  CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display --DUMP \
    TEST.IMS_PER_BATCH $NUM_GPU MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False OUTPUT_DIR TEMP  DATASETS.DATALOADER $DATALOADER DATASETS.DATALOADER_MODE $MODE \
    MODEL.META_ARCHITECTURE $TECHNIQUE TEST.PREDICT "GRAM_NOISE_COMPARE"
done

    









  
  
  
  


  
    
    
    

  





SUFFIX=VK
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
BACKBONE_TYPE=LORA
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX

DUMP_FOLDER=EXPS/$NAME
WT=$DUMP_FOLDER/model_best.pth
ls $WT 


CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST '("kitti_debug2",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE 

DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.751
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.848
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.669
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.834
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.767
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.700
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.833
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000


##### gt as predict  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000


  
  
SUFFIX=VK
TECHNIQUE=GeneralizedVLRCNN_PROPOSE
BACKBONE_TYPE=FAN
###### Entire MLP + Channel Attention 
BACKBONE_TYPE=FAN_MLP
###### Entire Network E2E 
BACKBONE_TYPE=FAN_E2E


NAME=GLIP-T_5-$BACKBONE_TYPE-VK/

DUMP_FOLDER=EXPS/$NAME
WT=$DUMP_FOLDER/model_best.pth
ls $WT 

PORT=12356

CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST '("kitti_debug2",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE 


CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST '("bdd100k_debug",)' MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE 









CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
OUTPUT=ucf_output/GLIP-T_5.txt

DATASET_NAME='("kitti_debug2",)'
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT tools/test_grounding_net.py --config-file $CONFIG --weight $WT --no-display \
  OUTPUT_DIR TEMP SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH $NUM_GPU \
  DATASETS.TEST $DATASET_NAME MODEL.DYHEAD.SCORE_AGG "MEAN" TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False 
  





###################### Training 

###################### Real World Weather 
CONFIG=configs/pretrain/glip_Swin_T_O365_GoldG.yaml
WT='MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'
MODEL_TYPE=GLIP-T_5
TRAIN_FN=multi_scale_train
TECHNIQUE=GeneralizedVLRCNN_PROPOSE


# ###### Virtual Kitty
Dataset_TRAIN='("kitti_train-grounding-debug",)'
Dataset_TEST='("kitti_debug2",)'
BATCHSIZE=2
SUFFIX=VK

BACKBONE_TYPE=LR_TKO

####### WORKS :: Starting the evaluations at 4650
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
  --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
  SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST SOLVER.MAX_ITER 100000 SOLVER.MAX_EPOCH 10000 \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True 
  

####### DOESNT WORK 
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-COCO_FINETUNE
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
  --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
  SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST SOLVER.MAX_ITER 100000 SOLVER.MAX_EPOCH 10000 \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
  SOLVER.BASE_LR 0.00001 SOLVER.LANG_LR 0.00001 SOLVER.STEPS \(0.67,0.89\) 


####### WORKS :: Starting the evaluations at 4700
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-FINETUNE
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
  --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
  SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST SOLVER.MAX_ITER 100000 SOLVER.MAX_EPOCH 10000 \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
  SOLVER.WEIGHT_DECAY 0.05 DATASETS.SHUFFLE_SEED 3 SOLVER.STEP_PATIENCE 3 SOLVER.AUTO_TERMINATE_PATIENCE 8 SOLVER.MODEL_EMA 0.0 SOLVER.TUNING_HIGHLEVEL_OVERRIDE full DATASETS.USE_CAPTION_PROMPT True 


####### Doesnt work  
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-LOSS
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
  --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
  SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST SOLVER.MAX_ITER 100000 SOLVER.MAX_EPOCH 10000 \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
  MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS True MODEL.DYHEAD.NUM_CLASSES 4 MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS True MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS True  
  


####### WORKS :: Starting the evaluations at 4700
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-LOSS2
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=0,1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
  --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
  SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST SOLVER.MAX_ITER 100000 SOLVER.MAX_EPOCH 10000 \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
  MODEL.DYHEAD.NUM_CLASSES 4 MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS True  


####### WORKS :: Starting the evaluations at 5150
NAME=$MODEL_TYPE-$BACKBONE_TYPE-$SUFFIX-LOSS3
OUTPUT_NOISE=ucf_output/$NAME
DUMP_FOLDER=EXPS/$NAME
CUDA_VISIBLE_DEVICES=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -W ignore -m torch.distributed.launch --nnodes 1 --nproc_per_node=$NUM_GPU --master_port=$PORT tools/train_net_proposed.py --config-file $CONFIG \
  --override_output_dir $DUMP_FOLDER --train_fn=$TRAIN_FN \
  SOLVER.IMS_PER_BATCH $((BATCHSIZE*NUM_GPU)) TEST.IMS_PER_BATCH $NUM_GPU TEST.DURING_TRAINING True MODEL.WEIGHT $WT TEST.EVAL_TASK detection MODEL.DYHEAD.SCORE_AGG "MEAN" MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
  DATASETS.TRAIN $Dataset_TRAIN DATASETS.TEST $Dataset_TEST SOLVER.MAX_ITER 100000 SOLVER.MAX_EPOCH 10000 \
  MODEL.NO_NORM True MODEL.META_ARCHITECTURE $TECHNIQUE MODEL.BACKBONE_TYPE $BACKBONE_TYPE DATALOADER.NUM_WORKERS 2 SOLVER.USE_AMP True \
  MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS True MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS True   
  
  

  

  

    
  
  
  



###### ToDO 
# https://openaccess.thecvf.com/content/ICCV2021/papers/Ramamonjison_SimROD_A_Simple_Adaptation_Method_for_Robust_Object_Detection_ICCV_2021_paper.pdf
 - Entorpy using sigmoid (un normalized) instead of softmax  
 - Dataset tsne + colors of 5 objects and hybrid 
 
# https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Vision-Language_Pre-Training_With_Triple_Contrastive_Learning_CVPR_2022_paper.pdf
# https://openaccess.thecvf.com//content/CVPR2022/papers/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.pdf
# https://arxiv.org/pdf/2306.15880 - Table 2 page 9

 - apply scene and action loss 
 







