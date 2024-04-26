export PYTHONPATH=$PWD/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=1 python val_fold1.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False
