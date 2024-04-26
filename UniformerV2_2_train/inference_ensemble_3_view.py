# https://github.com/MuhammadAsadJaved/Temporal-Activity-Detection-in-Untrimmed-Videos-with-Recurrent-Neural-Networks/blob/master/src/processing.py
import numpy as np
import os
import torch
import random
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.models import build_model
import cv2
import pandas as pd
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import glob
import warnings
from natsort import natsorted
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class VideoReader(object):
    def __init__(self, source):
        self.source = source
        try:
            self.source = int(source)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
        return was_read, frame

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()

@torch.no_grad()        
def main(cfg, videoids, checkpoint_list):
    print('Video ids:', videoids)
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[0]
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    print('Checkpoint 1:', cfg.TEST.CHECKPOINT_FILE_PATH)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[1]
    model_2 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_2)
    model_2.eval()
    print('Checkpoint 2:', cfg.TEST.CHECKPOINT_FILE_PATH)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[2]
    model_3 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_3)
    model_3.eval()
    print('Checkpoint 3:', cfg.TEST.CHECKPOINT_FILE_PATH)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[3]
    model_4 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_4)   
    model_4.eval() 
    print('Checkpoint 4:', cfg.TEST.CHECKPOINT_FILE_PATH)
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[4]
    model_5 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_5)   
    model_5.eval() 
    print('Checkpoint 5:', cfg.TEST.CHECKPOINT_FILE_PATH)

    total_prob_sq = {}
    video_order = []
    for key, values in videoids.items():
        video_order.append(values)
        video_path = values[1]
        print('Processing video:', video_path)
        img_provider = VideoReader(video_path)
        frames = []
        i = -1
        count = 0
        print('Num frames of the model:', cfg.DATA.NUM_FRAMES)
        print('Sampling rate of the model:', cfg.DATA.SAMPLING_RATE, '\n')
        prob_sq = []
        for able_to_read, frame in img_provider:
            count += 1
            i += 1
            if not able_to_read:
                break
            if len(frames) != cfg.DATA.NUM_FRAMES and count % cfg.DATA.SAMPLING_RATE == 0:
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_processed = cv2.resize(frame_processed, (336, 336), interpolation=cv2.INTER_AREA)
                frames.append(frame_processed)
            if len(frames) == cfg.DATA.NUM_FRAMES:
                # perform color normalization
                inputs = torch.tensor(np.array(frames)).float()
                inputs = inputs / 255.0
                inputs = inputs - torch.tensor(cfg.DATA.MEAN) # [0.45, 0.45, 0.45]
                inputs = inputs / torch.tensor(cfg.DATA.STD) # [0.225, 0.225,0.225]
                inputs = inputs.permute(3, 0, 1, 2) # [3, 8, 512, 512]
                inputs = inputs[None, :, :, :, :] # [1, 3, 8, 512, 512]
                inputs = [inputs]
                # transfer the data to the current GPU device
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                # ensemble models by views
                preds = model(inputs).detach().cpu().numpy()
                preds_2 = model_2(inputs).detach().cpu().numpy()
                preds_3 = model_3(inputs).detach().cpu().numpy()
                preds_4 = model_4(inputs).detach().cpu().numpy()
                preds_5 = model_5(inputs).detach().cpu().numpy()
                prob_ensemble = np.array([preds, preds_2, preds_3, preds_4, preds_5]) # [5, 1, 18]
                prob_ensemble = np.mean(prob_ensemble, axis=0) # [1, 18]
                prob_sq.append(prob_ensemble)
                frames = []
        total_prob_sq[values[0]] = prob_sq
    return dict(sorted(total_prob_sq.items())), video_order

def smoothing(x, k=3):
    # Applies a mean filter to an input sequence. The k value specifies the window size. window size = 2*k
    l = len(x) # 445
    s = np.arange(-k, l - k) # [-3,...,441]
    e = np.arange(k, l + k) # [3,...,447]
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape) # [445, 18]
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

def get_classification(sequence_class_prob):
    labels_index = np.argmax(sequence_class_prob, axis=1) # [445], returns list of position of max value in each list.
    probs = np.max(sequence_class_prob, axis=1)  # [445], return list of max value in each list
    return labels_index, probs
        
def activity_localization(prob_sq):
    action_idx, action_probs = get_classification(prob_sq)
    threshold = np.mean(action_probs)
    action_tag = np.zeros(action_idx.shape) # [445]
    action_tag[action_probs >= threshold] = 1
    activities_idx = []
    startings = []
    endings = []
    for i in range(len(action_tag)):
        if action_tag[i] == 1:
            activities_idx.append(action_idx[i])
            start = i
            end = i + 1
            startings.append(start)
            endings.append(end)
    return activities_idx, startings, endings

def merge_and_remove(data):
    df_total = pd.DataFrame([[0, 0, 0, 0]], columns=[0, 1, 2, 3])
    for i in range(1, 31):
        data_video = data[data[0] == i]
        list_label = data_video[1].unique()
        for label in list_label:
            data_video_label = data_video[data_video[1] == label]
            data_video_label = data_video_label.reset_index()
            for j in range(len(data_video_label) - 1):
                if data_video_label.loc[j + 1, 2] - data_video_label.loc[j, 3] <= 16: # merge two clips with same class
                    data_video_label.loc[j + 1, 2] = data_video_label.loc[j, 2]
                    data_video_label.loc[j, 3] = 0
                    data_video_label.loc[j, 2] = 0
            df_total = df_total.append(data_video_label)
    df_total = df_total[df_total[3] != 0]
    df_total = df_total[df_total[3] - df_total[2] > 6] # remove action segments
    df_total = df_total.drop(columns=['index'])
    df_total = df_total.sort_values(by=[0, 1])
    df_total.to_csv('A2_submission.txt', sep=' ', index=False, header=False)

def general_submission(data):
    data_filtered = data[data[1] != 0]
    data_filtered[2] = data[2].map(lambda x: int(float(x)))
    data_filtered[3] = data[3].map(lambda x: int(float(x)))
    data_filtered = data_filtered.sort_values(by=[0, 1])
    merge_and_remove(data_filtered)

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    fps = 30
    seed_everything(719)
    labels = list(np.arange(16)) # change to 16 classes for dataset 2023 and 2024
    checkpoint_dashboard_list = []
    checkpoint_rearview_list = []
    checkpoint_right_list = []
    path_saved_ckpt = natsorted(glob.glob('checkpoint_submit/*.pyth'))
    for i in range(len(path_saved_ckpt)):
        if path_saved_ckpt[i].split('/')[-1].split('_')[0] == 'dash':
            checkpoint_dashboard_list.append(path_saved_ckpt[i])
        elif path_saved_ckpt[i].split('/')[-1].split('_')[0] == 'rear':
            checkpoint_rearview_list.append(path_saved_ckpt[i])
        else:
            checkpoint_right_list.append(path_saved_ckpt[i])

    # start infer dashboard videos
    video_ids = {}
    video_names = []
    path = cfg.DATA.PATH_TO_DATA_DIR
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = pd.read_csv(csvfile, delimiter=',')
        for idx, row in csvReader.iterrows():
            video_ids[row[1]] = row[0]
            video_names.append(row[1])
    text_files = glob.glob(path + "/**/*.MP4", recursive=True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:
            if vid_name in video_names:
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids:
            vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist:
            vid_info.setdefault(key, []).append(filelist[key])
    vid_info = dict(sorted(vid_info.items()))
    prob_1, video_order = main(cfg, vid_info, checkpoint_dashboard_list)

    # start infer rearview videos
    video_ids = {}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = pd.read_csv(csvfile, delimiter=',')
        for idx, row in csvReader.iterrows():
            video_ids[row[2]] = row[0]
            video_names.append(row[2])
    text_files = glob.glob(path + "/**/*.MP4", recursive = True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:
            if vid_name in video_names:
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids:
            vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist:
            vid_info.setdefault(key, []).append(filelist[key])
    vid_info = dict(sorted(vid_info.items()))
    prob_2, video_order = main(cfg, vid_info, checkpoint_rearview_list)

    # start infer rightside videos
    video_ids = {}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = pd.read_csv(csvfile, delimiter=',')
        for idx, row in csvReader.iterrows():
            video_ids[row[3]] = row[0]
            video_names.append(row[3])
    text_files = glob.glob(path + "/**/*.MP4", recursive = True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:
            if vid_name in video_names:
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids:
            vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist:
            vid_info.setdefault(key, []).append(filelist[key])
    vid_info = dict(sorted(vid_info.items()))
    prob_3, video_order = main(cfg, vid_info, checkpoint_right_list)

    # post-processing
    prob_ensemble = []
    dataframe_list = []
    for i in range(1, len(vid_info) + 1): # 11
        len_prob = min(len(prob_1[i]), len(prob_2[i]), len(prob_3[i])) # 445
        prob_ensemble_video = []
        for ids in range(len_prob):
            prob_sub_mean = (prob_1[i][ids] + prob_2[i][ids] + prob_3[i][ids])/3
            prob_ensemble_video.append(prob_sub_mean)
        # classification step
        prob_actions = np.array(prob_ensemble_video) # [445, 1, 18]
        prob_actions = np.squeeze(prob_actions) # [445, 18]
        # temporal localization step
        prediction_smoothed = smoothing(prob_actions, 3)
        activities_idx, startings, endings = activity_localization(prob_actions)
        for idx, s, e in zip(activities_idx, startings, endings):
            start = s * float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / fps
            end = e * float(cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE) / fps
            label = labels[idx]
            dataframe_list.append([i, label, start, end])
    data = pd.DataFrame(dataframe_list, columns=[0, 1, 2, 3])
    general_submission(data)
