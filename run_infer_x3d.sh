cd X3D_train
export PYTHONPATH=$PWD/:$PYTHONPATH
python3 my_post_process.py --cfg configs/X3D_L.yaml NUM_GPUS 1 TRAIN.ENABLE False
cd ..
