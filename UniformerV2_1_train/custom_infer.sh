export PYTHONPATH=$PWD/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python custom_infer/generate_prob.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR A2
# python custom_infer/infer_result.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False DATA.PATH_TO_DATA_DIR A2
