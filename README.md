# This is the solution for track 3 of the AI City Challenge 2024

## 1. Setup
### 1.1 Run from conda
conda env create -f environment.yml

conda activate anomaly

### 1.2 Run from Docker
To be released.

## 2. Dataset preparation
To get cut videos for training, please download it from this [link](https://drive.google.com/file/d/13HEJptRQeu_0yzmX8NsRr4qdqgAaY4jZ/view?usp=sharing). After downloading, extract the file and put it to three folders X3D_train/data, VideoMAE_train/data, and UniformerV2_1_train/data.

To get custom cut videos for training UniformerV2_2, please download it from this [link](https://drive.google.com/file/d/1HFxKcY0RFh1UJBH00PAHOqrlhI6w8UX9/view?usp=sharing). After downloading, extract the file and put it to folder UniformerV2_2_train/data.

## 3. Dataset structure
### 3.1 X3D
The AI City dataset is organized with the following structure:
```
X3D_train
|_ A2
   |_ user_id_12670
   |  |_ *.mp4
   |_ user_id_13148
   |  |_ *.mp4
   |_ ...
   |  |_ *.mp4
   |_ user_id_96715
      |_ *.mp4
|_ data
   |_ A1_clip
      |_ 0
      |  |_ *.mp4
      |_ 1
      |  |_ *.mp4
      |_ ...
      |  |_ *.mp4
      |_ 15
         |_ *.mp4
```

### 3.2 UniformerV2_1
To be released.

### 3.3 UniformerV2_2
```
UniformerV2_2_train
|_ A2
   |_ user_id_12670
   |  |_ *.mp4
   |_ user_id_13148
   |  |_ *.mp4
   |_ ...
   |  |_ *.mp4
   |_ user_id_96715
      |_ *.mp4
|_ data
   |_ A1_clip_custom
      |_ 0
      |  |_ *.mp4
      |_ 1
      |  |_ *.mp4
      |_ 2
      |  |_ *.mp4
      |_ 3
         |_ *.mp4
```

### 3.4 VideoMAE
```
VideoMAE_train
|_ A1
   |_ user_id_13522
   |  |_ *.mp4
   |_ user_id_14786
   |  |_ *.mp4
   |_ ...
   |  |_ *.mp4
   |_ user_id_99882
      |_ *.mp4
|_ data
   |_ A1_clip
      |_ 0
      |  |_ *.mp4
      |_ 1
      |  |_ *.mp4
      |_ ...
      |  |_ *.mp4
      |_ 15
         |_ *.mp4
|_ A2
   |_ user_id_12670
   |  |_ *.mp4
   |_ user_id_13148
   |  |_ *.mp4
   |_ ...
   |  |_ *.mp4
   |_ user_id_96715
      |_ *.mp4
```

## 4. Usage
### 4.1 For conda
#### 4.1.1 X3D
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

#### 4.1.2 UniformerV2_1
To be released.

#### 4.1.3 UniformerV2_2
To train UniformerV2_2, follow the code snippets bellow:
```bash
cd UniformerV2_2
# Step 1: Train UniformerV2_2
bash train.sh
# Step 2: Rename and move checkpoints
python move_ckpt.py
# Step 3: Infer UniformerV2_2
bash infer.sh
```

#### 4.1.4 VideoMAE
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

### 4.2 For docker
To be released.

### 5. Ensemble model
To ensemble four models, follow the code snippets bellow:
```bash
cd infer
python run_submission_ensemble.py
```

## 6. Citation
If you find our work useful, please cite the following:
```
@inproceedings{nguyen2024multi,
  title={Multi-View Spatial-Temporal Learning for Understanding Unusual Behaviors in Untrimmed Naturalistic Driving Videos},
  author={Huy-Hung Nguyen, Chi Dai Tran, Long Hoang Pham, Duong Nguyen-Ngoc Tran, Tai Huu-Phuong Tran, Duong Khac Vu, Quoc Pham-Nam Ho, Ngoc Doan-Minh Huynh, Huyng-Min Jeon, Hyung-Joon Jeon, Jae Wook Jeon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={x--x},
  year={2024}
}
```

## 7. Contact
If you have any questions, feel free to contact 'Huy-Hung Nguyen' ([huyhung411991@gmail.com](huyhung411991@gmail.com)), or `Chi Dai Tran` ([ctran743@gmail.com](ctran743@gmail.com)).

##  8. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [VTCC-uTVM/2022AICityChallenge-Track3](https://github.com/VTCC-uTVM/2022AICityChallenge-Track3)
* [Meituan-IoTCV/aicity_release](https://github.com/Meituan-IoTCV/aicity_release)
* [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)
* [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
<!--te-->