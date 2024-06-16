# python3 generate_test_ids.py --test_path B/

# cd X3D_train
# export PYTHONPATH=$PWD/:$PYTHONPATH
# python3 my_post_process.py --cfg configs/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False
# cd ..

# cd UniformerV2_1_train
# export PYTHONPATH=$PWD/:$PYTHONPATH
# python3 my_post_process.py --cfg exp/k710/k710_l14_f8x336/config.yaml NUM_GPUS 1 TRAIN.ENABLE False
# cd ..

# cd UniformerV2_2_train
# export PYTHONPATH=$PWD/:$PYTHONPATH
# python3 my_post_process.py --cfg my_config.yaml NUM_GPUS 1 TRAIN.ENABLE False
# cd ..

# cd VideoMAE_train
# bash scripts/cls/inference_cls.sh
# cd ..

rm -rf infer/pickle_uniformerv2_full
rm -rf infer/pickle_uniformerv2_4cls
rm -rf infer/pickle_x3d
rm -rf infer/pickle_videomae
cp -r UniformerV2_1_train/pickle_uniformerv2_full infer/pickle_uniformerv2_full
cp -r UniformerV2_2_train/pickle_uniformerv2_4cls infer/pickle_uniformerv2_4cls
cp -r X3D_train/pickle_x3d infer/pickle_x3d
cp -r VideoMAE_train/pickle_videomae infer/pickle_videomae
cd infer
python3 run_submission_ensemble.py
cd ..
rm -rf infer/pickle_uniformerv2_full
rm -rf infer/pickle_uniformerv2_4cls
rm -rf infer/pickle_x3d
rm -rf infer/pickle_videomae
