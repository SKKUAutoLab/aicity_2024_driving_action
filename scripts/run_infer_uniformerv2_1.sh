python3 generate_test_ids.py --test_path B/
cd UniformerV2_1_train
export PYTHONPATH=$PWD/:$PYTHONPATH
python3 my_post_process.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False
cd ..