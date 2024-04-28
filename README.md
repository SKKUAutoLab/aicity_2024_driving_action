# [CVPRW 2024] Multi-View Spatial-Temporal Learning for Understanding Unusual Behaviors in Untrimmed Naturalistic Driving Videos

This repository contains the source code for AI City Challenge 2024 Track 3 (Naturalistic Driving Action Recognition).

- Team Name: SKKU-AutoLab 
- Team ID: 05

## 1. Setup
### 1.1 Run from conda (for both training and inference)
#### Using environment.yml
conda env create --name track3 --file=environment.yml

conda activate track3

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

#### Using requirements.txt
conda create --name track3 python=3.10.13

conda activate track3

pip install -r requirements.txt

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install detectron2-0.6-cp310-cp310-linux_x86_64.whl

### 1.2 Run from Docker (only for inference)
sudo docker load < docker_aic24_track3_final.tar

docker run --ipc=host --gpus all -v <LOCAL_SOURCE_CODE>:/usr/src/aic24-track_3/ \
                                 -v <LOCAL_INPUT_DATA>:/usr/src/aic24-track_3/B/ \
							     -v <LOCAL_OUTPUT_FOLDER>:/usr/src/aic24-track_3/output_submission/ \
					             -it <IMAGE_ID>

Ex: docker run --ipc=host --gpus all -v /home/vsw/Downloads/AIC24-Track03/:/usr/src/aic24-track_3/ \
								     -v /home/vsw/Downloads/B/:/usr/src/aic24-track_3/B/ \
									 -v /home/vsw/Downloads/output_submission/:/usr/src/aic24-track_3/output_submission/ \
							         -it 96f8bfc76877

## 2. Dataset preparation
To get cut videos for training X3D, UniformerV1_1, and VideoMAE, please download it from this [link](https://drive.google.com/file/d/13HEJptRQeu_0yzmX8NsRr4qdqgAaY4jZ/view?usp=sharing). After downloading, extract the file and put it to three folders X3D_train/data, VideoMAE_train/data/A1_clip (only put sub folders in the A1_clip folder), and UniformerV2_1_train/data.

To get custom cut videos for training UniformerV2_2, please download it from this [link](https://drive.google.com/file/d/1HFxKcY0RFh1UJBH00PAHOqrlhI6w8UX9/view?usp=sharing). After downloading, extract the file and put it to folder UniformerV2_2_train/data.

To get pretrained backbone weights for UniformerV2_1 and UniformerV2_2, please download it from this [link](https://drive.google.com/file/d/10mCzuJZCUSqrkVpv8An8nzCJUfMj8skl/view?usp=sharing) and this [link](https://drive.google.com/file/d/1uSCu22RMYIh7x7phB1HcGNQ9HsXSq-Fx/view?usp=sharing). After downloading, extract the file and put it to two folders UniformerV2_1 and UniformerV2_2. 

To get pretrained weights for VideoMAE, please download it from this [link](https://drive.google.com/file/d/1Fr7e_Q49o-Ug5VlasrZfp8NUVPw-AdF8/view?usp=sharing). After downloading, extract the file and put it to the folder VideoMAE_train.

To get docker file to make an inference on a custom dataset, please download it from this [link](https://drive.google.com/file/d/10_wLt8mK3QoCmQjnkOOvEiv7lPbjUD_w/view?usp=sharing).

## 3. Weight preparation (only use to infer from source)
To get X3D weights, please download them from this [link](https://drive.google.com/file/d/1TpcfCkKSMhPjyHqbsopQjl7I9fYMvFDE/view?usp=sharing). After downloading, extract the file and put it to the folder X3D_train.

To get UniformerV2_1 weights, please download them from this [link](https://drive.google.com/file/d/1eONE4evmZ2smmjgp2q-N4UcWWiib3pDd/view?usp=sharing). After downloading, extract the file and put it to the folder UniformerV2_1_train.

To get UniformerV2_2 weights, please download them from this [link](https://drive.google.com/file/d/1rK7lhVGpyRlqDLZFz5_GWrWUyCdZQzJg/view?usp=sharing). After downloading, extract the file and put it to the folder UniformerV2_2_train.

To get VideoMAE weights, please download them from this [link](https://drive.google.com/file/d/14KZPd5kHw0kNzQZi0DFF-8AfXeE3DSZa/view?usp=sharing). After downloading, extract the file and put it to the folder VideoMAE_train.

## 3. Dataset structure
### 3.1 X3D
For X3D model, the dataset is organized with the following structure:
```
X3D_train
|_ data
|  |_ A1_clip
|  |  |_ 0
|  |  |  |_ *.mp4
|  |  |_ 1
|  |  |  |_ *.mp4
|  |  |_ ...
|  |  |  |_ *.mp4
|  |  |_ 15
|  |  |  |_ *.mp4
|  |_ *.csv
|_ pickle_x3d
|  |_ A2
|  |  |_ *.pkl
|_ checkpoint_x3d
|  |_ *.pyth
```

### 3.2 UniformerV2_1
For UniformerV2_1 model, the dataset is organized with the following structure:
```
UniformerV2_1_train
|_ A2
|  |_ user_id_12670
|  |  |_ *.mp4
|  |_ user_id_13148
|  |  |_ *.mp4
|  |_ ...
|  |  |_ *.mp4
|  |_ user_id_96715
|  |  |_ *.mp4
|_ data
|  |_ A1_clip
|  |  |_ 0
|  |  |  |_ *.mp4
|  |  |_ 1
|  |  |  |_ *.mp4
|  |  |_ ...
|  |  |  |_ *.mp4
|  |  |_ 15
|  |  |  |_ *.mp4
|_ pickle_uniformerv2_full
|  |_ *.pkl
|_ checkpoint_uniformerv2_full
|  |_ *.pyth
|_ k710_uniformerv2_l14_8x336.pyth
|_ vit_saved
|  |  |_ vit_b16.pth
|  |  |_ vit_l14.pth
|  |  |_ vit_l14_336.pth
```

### 3.3 UniformerV2_2
For UniformerV2_2 model, the dataset is organized with the following structure:
```
UniformerV2_2_train
|_ A2
|  |_ user_id_12670
|  |  |_ *.mp4
|  |_ user_id_13148
|  |  |_ *.mp4
|  |_ ...
|  |  |_ *.mp4
|  |_ user_id_96715
|  |  |_ *.mp4
|_ data
|  |_ A1_clip_custom
|  |  |_ 0
|  |  |  |_ *.mp4
|  |  |_ 1
|  |  |  |_ *.mp4
|  |  |_ 2
|  |  |  |_ *.mp4
|  |  |_ 3
|  |  |  |_ *.mp4
|_ pickle_uniformerv2_4lcs
|  |_ *.pkl
|_ checkpoint_uniformerv2_4cls
|  |_ *.pyth
|_ k710_uniformerv2_l14_8x336.pyth
|_ vit_saved
|  |  |_ vit_b16.pth
|  |  |_ vit_l14.pth
|  |  |_ vit_l14_336.pth
```

### 3.4 VideoMAE
For VideoMAE model, the dataset is organized with the following structure:
```
VideoMAE_train
|_ data
|  |_ A1_clip
|  |  |_ 0
|  |  |  |_ *.mp4
|  |  |_ 1
|  |  |  |_ *.mp4
|  |  |_ ...
|  |  |  |_ *.mp4
|  |  |_ 15
|  |  |  |_ *.mp4
|  |  |_ *.csv
|_ pretrained_models
|  |_ vit_l_hybrid_pt_800e_k700_ft.pth
```

## 4. Usage
### 4.1 X3D
To train X3D, follow the code snippets bellow:
```bash
cd X3D_train
# Step 1: Train X3D
bash train.sh
# Step 2: Rename and move checkpoints
python move_ckpt.py
cd ..
```

#### 4.2 UniformerV2_1
To train UniformerV2_1, follow the code snippets bellow:
```bash
cd UniformerV2_1_train
# Step 1: Train UniformerV2_1
bash train.sh
# Step 2: Rename and move checkpoints
python move_ckpt.py
cd ..
```

#### 4.3 UniformerV2_2
To train UniformerV2_2, follow the code snippets bellow:
```bash
cd UniformerV2_2_train
# Step 1: Train UniformerV2_2
bash train.sh
# Step 2: Rename and move checkpoints
python move_ckpt.py
cd ..
```

#### 4.4 VideoMAE
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
cd ..
```

### 5. Ensemble model
To ensemble four models, run the following script:
```bash
bash run_infer_all.sh
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
If you have any questions, feel free to contact `Huy H. Nguyen` ([huyhung411991@gmail.com](huyhung411991@gmail.com)), or `Chi D. Tran` ([ctran743@gmail.com](ctran743@gmail.com)).

##  8. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [VTCC-uTVM/2022AICityChallenge-Track3](https://github.com/VTCC-uTVM/2022AICityChallenge-Track3)
* [Meituan-IoTCV/aicity_release](https://github.com/Meituan-IoTCV/aicity_release)
* [OpenGVLab/UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)
* [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
<!--te-->