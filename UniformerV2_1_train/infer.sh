export PYTHONPATH=$PWD/:$PYTHONPATH
python my_post_process.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False
# python my_submission.py
