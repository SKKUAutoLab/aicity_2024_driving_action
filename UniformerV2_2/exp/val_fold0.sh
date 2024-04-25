export PYTHONPATH=$PWD/:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python val_fold0.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False
