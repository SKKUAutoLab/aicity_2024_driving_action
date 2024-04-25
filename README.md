# This is the solution for track 3 of the AI City Challenge 2024

## Setup
### Run from conda
conda env create -f environment.yml

conda activate anomaly

### Run from Docker
To be released.

## Dataset preparation
To get cut videos for training, please download it from this link ... 

## Usage
### X3D
To train X3D, follow the code snippets bellow:
```bash
cd X3D_train
# Step 1: Generate video_ids for inference
cd A2
python generate_test_ids.py --test_path A2/
cd ..
# Step 2: Train X3D
bash train.sh
# Step 3: Rename and move checkpoints
python move_ckpt.py
# Step 4: Infer X3D
bash infer.sh
```

### UniformerV2_1
To be released.

### UniformerV2_2
To be released.

### VideoMAE
To train VideoMAE, follow the code snippets bellow:
```bash
cd VideoMAE_train
# Step 1: Train VideoMAE
bash scripts/cls/train_fold0.sh
bash scripts/cls/train_fold1.sh
bash scripts/cls/train_fold2.sh
bash scripts/cls/train_fold3.sh
bash scripts/cls/train_fold4.sh
# Step 2: Rename and move checkpoints
python move_ckpt.py
# Step 3: Infer VideoMAE
bash scripts/cls/inference_cls.sh
```

### Ensemble
To ensemble four models, follow the code snippets bellow:
```bash
cd infer
python run_submission_ensemble.py
```

## Citation
If you find our work useful, please cite the following:

## Contact
If you have any questions, feel free to contact 'Huy-Hung Nguyen' ([huyhung411991@gmail.com](huyhung411991@gmail.com)), or `Chi Dai Tran` ([ctran743@gmail.com](ctran743@gmail.com)).

##  Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [VTCC-uTVM/2022AICityChallenge-Track3](https://github.com/VTCC-uTVM/2022AICityChallenge-Track3)
* [Meituan-IoTCV/aicity_release](https://github.com/Meituan-IoTCV/aicity_release)
* [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)
* [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
<!--te-->