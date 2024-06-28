python3 generate_test_ids.py --test_path B/
cd UniformerV2_2_train
export PYTHONPATH=$PWD/:$PYTHONPATH
python3 my_post_process.py --cfg my_config.yaml NUM_GPUS 1 TRAIN.ENABLE False
cd ..