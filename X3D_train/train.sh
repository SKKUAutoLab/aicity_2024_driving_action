export PYTHONPATH=$PWD/:$PYTHONPATH
python -W ignore tools/run_net.py --cfg configs/X3D_L.yaml NUM_GPUS 4 DATA.PATH_TO_DATA_DIR data
