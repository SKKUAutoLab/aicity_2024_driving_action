OUTPUT_DIR='pickle_videomae/A2'
DATA_PATH='../B/'

MODEL_PATH='checkpoint_videomae/dash_0.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "dash" \
    --clip_stride 30 \
    --fold 0 \
    --crop \

MODEL_PATH='checkpoint_videomae/dash_1.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "dash" \
    --clip_stride 30 \
    --fold 1 \
    --crop \


MODEL_PATH='checkpoint_videomae/dash_2.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "dash" \
    --clip_stride 30 \
    --fold 2 \
    --crop \


MODEL_PATH='checkpoint_videomae/dash_3.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "dash" \
    --clip_stride 30 \
    --fold 3 \
    --crop \

MODEL_PATH='checkpoint_videomae/dash_4.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "dash" \
    --clip_stride 30 \
    --fold 4 \
    --crop \

MODEL_PATH='checkpoint_videomae/right_0.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "right" \
    --clip_stride 30 \
    --fold 0 \
    --crop \

MODEL_PATH='checkpoint_videomae/right_1.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "right" \
    --clip_stride 30 \
    --fold 1 \
    --crop \

MODEL_PATH='checkpoint_videomae/right_2.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "right" \
    --clip_stride 30 \
    --fold 2 \
    --crop \

MODEL_PATH='checkpoint_videomae/right_3.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "right" \
    --clip_stride 30 \
    --fold 3 \
    --crop \

MODEL_PATH='checkpoint_videomae/right_4.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "right" \
    --clip_stride 30 \
    --fold 4 \
    --crop \

MODEL_PATH='checkpoint_videomae/rear_0.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "rear" \
    --clip_stride 30 \
    --fold 0 \
    --crop \


MODEL_PATH='checkpoint_videomae/rear_1.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "rear" \
    --clip_stride 30 \
    --fold 1 \
    --crop \


MODEL_PATH='checkpoint_videomae/rear_2.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "rear" \
    --clip_stride 30 \
    --fold 2 \
    --crop \



MODEL_PATH='checkpoint_videomae/rear_3.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "rear" \
    --clip_stride 30 \
    --fold 3 \
    --crop \



MODEL_PATH='checkpoint_videomae/rear_4.pth'
# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs) 
CUDA_VISIBLE_DEVICES=0 python evaluate_loc.py \
    --model vit_large_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 16 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224\
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt lion \
    --opt_betas 0.9 0.99 \
    --weight_decay 0.05 \
    --epochs 35 \
    --lr 2e-3 \
    --view "rear" \
    --clip_stride 30 \
    --fold 4 \
    --crop \
