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
from natsort import natsorted
import warnings
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
def inference_result(cfg, videoids, checkpoint_list):
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    for checki in checkpoint_list:
        cfg.TEST.CHECKPOINT_FILE_PATH = checki
        viewtype = checki.split('/')[-1].split('_')[0]
        model = build_model(cfg)
        cu.load_test_checkpoint(cfg, model)
        model.eval()
        print('Load checkpoint:', cfg.TEST.CHECKPOINT_FILE_PATH, '\n')
        total_prob_sq1 = {}
        total_prob_sq2 = {}
        total_prob_sq3 = {}
        total_prob_sq4 = {}
        video_order = []
        for key, values in videoids.items():
            video_order.append(values)
            video_path = values[1]
            print('Processing video:', video_path)
            img_provider = VideoReader(video_path)
            frames = []
            frames2 = []
            frames3 = []
            frames4 = []
            count = 0
            count2 = 0
            count3 = 0
            count4 = 0
            prob_sq1 = []
            prob_sq2 = []
            prob_sq3 = []
            prob_sq4 = []
            for able_to_read, frame in img_provider:
                count += 1
                if not able_to_read:
                    break
                if len(frames) != cfg.DATA.NUM_FRAMES and count % cfg.DATA.SAMPLING_RATE == 0 and count > 0 * cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE:
                    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed = cv2.resize(frame_processed, (336, 336), interpolation=cv2.INTER_AREA)
                    frames.append(frame_processed)
                if len(frames) == cfg.DATA.NUM_FRAMES:
                    # perform color normalization
                    inputs = torch.tensor(np.array(frames)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    inputs = inputs.permute(3, 0, 1, 2)
                    inputs = inputs[None, :, :, :, :]
                    inputs = [inputs]
                    # transfer the data to the current GPU device
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    # ensemble models by views
                    preds = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)
                    for i in range(cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE):
                        prob_sq1.append(prob_ensemble)
                    frames = []

            for able_to_read, frame in img_provider:
                count2 += 1
                if not able_to_read:
                    break
                if len(frames2) != cfg.DATA.NUM_FRAMES and count2 % cfg.DATA.SAMPLING_RATE == 0 and count2 > 0.25 * cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE:
                    frame_processed2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed2 = cv2.resize(frame_processed2, (336, 336), interpolation=cv2.INTER_AREA)
                    frames2.append(frame_processed2)
                if len(frames2) == cfg.DATA.NUM_FRAMES:
                    # perform color normalization
                    inputs = torch.tensor(np.array(frames2)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    inputs = inputs.permute(3, 0, 1, 2)
                    inputs = inputs[None, :, :, :, :]
                    inputs = [inputs]
                    # transfer the data to the current GPU device
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    preds_2 = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds_2])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)
                    for i in range(cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE):
                        prob_sq2.append(prob_ensemble)
                    frames2 = []

            for able_to_read, frame in img_provider:
                count3 += 1
                if not able_to_read:
                    break
                if len(frames3) != cfg.DATA.NUM_FRAMES and count3 % cfg.DATA.SAMPLING_RATE == 0 and count3 > 0.5 * cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE:
                    frame_processed3 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed3 = cv2.resize(frame_processed3, (336, 336), interpolation=cv2.INTER_AREA)
                    frames3.append(frame_processed3)
                if len(frames3) == cfg.DATA.NUM_FRAMES:
                    # perform color normalization
                    inputs = torch.tensor(np.array(frames3)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    inputs = inputs.permute(3, 0, 1, 2)
                    inputs = inputs[None, :, :, :, :]
                    inputs = [inputs]
                    # transfer the data to the current GPU device
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    preds_3 = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds_3])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)
                    for i in range(cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE):
                        prob_sq3.append(prob_ensemble)
                    frames3 = []

            for able_to_read, frame in img_provider:
                count4 += 1
                if not able_to_read:
                    break
                if len(frames4) != cfg.DATA.NUM_FRAMES and count4 % cfg.DATA.SAMPLING_RATE == 0 and count3 > 0.75 * cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE:
                    frame_processed4 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_processed4 = cv2.resize(frame_processed4, (336, 336), interpolation=cv2.INTER_AREA)
                    frames4.append(frame_processed4)
                if len(frames4) == cfg.DATA.NUM_FRAMES:
                    # perform color normalization
                    inputs = torch.tensor(np.array(frames4)).float()
                    inputs = inputs / 255.0
                    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                    inputs = inputs / torch.tensor(cfg.DATA.STD)
                    inputs = inputs.permute(3, 0, 1, 2)
                    inputs = inputs[None, :, :, :, :]
                    inputs = [inputs]
                    # transfer the data to the current GPU device
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    preds_4 = model(inputs).detach().cpu().numpy()
                    prob_ensemble = np.array([preds_4])
                    prob_ensemble = np.mean(prob_ensemble, axis=0)
                    for i in range(cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE):
                        prob_sq4.append(prob_ensemble)
                    frames4 = []

            total_prob_sq1[values[0]] = prob_sq1
            total_prob_sq2[values[0]] = prob_sq2
            total_prob_sq3[values[0]] = prob_sq3
            total_prob_sq4[values[0]] = prob_sq4
        np.save('probability_results/view_{}_frame_{}_rate_{}_overlapratio_{}.npy'.format(viewtype, cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE, 0), dict(sorted(total_prob_sq1.items())))
        np.save('probability_results/view_{}_frame_{}_rate_{}_overlapratio_{}.npy'.format(viewtype, cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE, 0.25), dict(sorted(total_prob_sq2.items())))
        np.save('probability_results/view_{}_frame_{}_rate_{}_overlapratio_{}.npy'.format(viewtype, cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE, 0.5), dict(sorted(total_prob_sq3.items())))
        np.save('probability_results/view_{}_frame_{}_rate_{}_overlapratio_{}.npy'.format(viewtype, cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE, 0.75), dict(sorted(total_prob_sq4.items())))

if __name__ == "__main__":
    if not os.path.exists('probability_results'):
        os.makedirs('probability_results')
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    fps = 30
    seed_everything(719)
    labels = list(np.arange(16)) # 18 classes for dataset 2021
    checkpoint_dashboard_list = []
    checkpoint_rearview_list = []
    checkpoint_right_list = []
    path_saved_ckpt = natsorted(glob.glob('checkpoint_submit/*.pyth'))
    for i in range(len(path_saved_ckpt)):
        if path_saved_ckpt[i].split('/')[-1].split('_')[0] == 'dashboard':
            checkpoint_dashboard_list.append(path_saved_ckpt[i])
        elif path_saved_ckpt[i].split('/')[-1].split('_')[0] == 'rearview':
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
    inference_result(cfg, vid_info, checkpoint_dashboard_list)

    # start infer rearview videos
    video_ids = {}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = pd.read_csv(csvfile, delimiter=',')
        for idx, row in csvReader.iterrows():
            video_ids[row[2]] = row[0]
            video_names.append(row[2])
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
    inference_result(cfg, vid_info, checkpoint_rearview_list)

    # start infer rightside videos
    video_ids = {}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = pd.read_csv(csvfile, delimiter=',')
        for idx, row in csvReader.iterrows():
            video_ids[row[3]] = row[0]
            video_names.append(row[3])
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
    inference_result(cfg, vid_info, checkpoint_right_list)
