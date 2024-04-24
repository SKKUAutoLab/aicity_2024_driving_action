from torch.utils.data import Dataset
import pandas as pd
import re
import os
from decord import VideoReader, cpu
import numpy as np
import video_transforms as video_transforms
import volume_transforms as volume_transforms
import torch
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import assert_and_infer_cfg
import random
import torch.nn.functional as F
import pickle
from natsort import natsorted
import glob

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class VideoInferDataset(Dataset):
    def __init__(self, data_root, video_csv_path, clip_len=8, frame_sample_rate=4, clip_stride=30, short_side_size=512,
                 new_height=512, new_width=512, keep_aspect_ratio=True, crop=False, view="dashboard"):
        self.videos = pd.read_csv(video_csv_path)
        self.crop = crop
        if self.crop:
            print("Cut off useless region")
        else:
            print("No cut off useless region")
        data = dict()
        data["dash"] = []
        data["rear"] = []
        data["right"] = []
        for idx, row_data in self.videos.iterrows():
            user_id = re.search("user_id_\d{5}", row_data[1])[0]
            data["dash"].append(os.path.join(data_root, user_id, row_data[1]))
            data["rear"].append(os.path.join(data_root, user_id, row_data[2]))
            data["right"].append(os.path.join(data_root, user_id, row_data[3]))
        data_list = sorted(data[view])
        self.data_samples = []
        self.start_frame_idxs = []
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.view = view
        self.short_side_size = short_side_size
        self.clip_stride = clip_stride # 30 fps
        for data_path in data_list:
            vr = VideoReader(data_path, num_threads=1, ctx=cpu(0))
            for start_idx in range(0, len(vr) - 1, self.clip_stride):
                self.data_samples.append(data_path)
                self.start_frame_idxs.append(start_idx) # [0,30,...,14250]
        self.data_resize = video_transforms.Compose([video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear')])
        self.data_transform = video_transforms.Compose([volume_transforms.ClipToTensor(), video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        sample = self.data_samples[index]
        start_frame_idx = self.start_frame_idxs[index]
        t_index = start_frame_idx // self.clip_stride
        buffer = self.loadvideo_decord(sample, start_frame_idx=start_frame_idx) # [8, 1080, 1920, 3]
        if self.crop:
            buffer = buffer[:, 128:, 128:, :]
        buffer = self.data_resize(buffer) # [512, 512, 3], len(8)
        buffer = self.data_transform(buffer) # [3, 8, 512, 512]
        return buffer, sample.split("/")[-1].split(".")[0], t_index

    def loadvideo_decord(self, sample, start_frame_idx=0, sample_rate_scale=1):
        fname = sample
        if not (os.path.exists(fname)):
            return []
        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height, num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []
        all_index = []
        converted_len = int(self.clip_len * self.frame_sample_rate) # 32
        if start_frame_idx > len(vr) - 1: # [14259]
            all_index = [len(vr) - 1] * self.clip_len
        else:
            end_idx = start_frame_idx + converted_len
            index = np.linspace(start_frame_idx, end_idx, num=self.clip_len)
            index = np.clip(index, start_frame_idx, len(vr) - 1).astype(np.int64)
            all_index.extend(list(index))
            all_index = all_index[::int(sample_rate_scale)] # [8]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy() # [8, 1080, 1920, 3]
        return buffer

    def __len__(self):
        return len(self.data_samples) # (14295/clip_stride) * 2 user_id

def main(cfg, data_path, csv_file, sampling_rate, clip_stride, crop, view, fold, checkpoint_file_path):
    video_csv = os.path.join(data_path, csv_file)
    dataset = VideoInferDataset(data_path, video_csv, frame_sample_rate=sampling_rate, clip_stride=clip_stride, crop=crop, view=view)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8)
    row_data = []
    # load model
    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_file_path
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    # start inference
    for idx, batch in enumerate(test_loader):
        if idx % 10 == 0:
            print("Processed {} batch".format(idx))
        input_data = batch[0] # [8, 3, 8, 512, 512]
        input_data = [input_data]
        if isinstance(input_data, (list,)):
            for i in range(len(input_data)):
                input_data[i] = input_data[i].cuda(non_blocking=True)
        else:
            input_data = input_data.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input_data) # [8, 18]
            probs = F.softmax(logits, dim=-1) # [8, 18]
            row_data.extend(zip(list(batch[1]), batch[2].cpu().numpy().tolist(), probs.cpu().numpy().tolist()))
    results = pd.DataFrame(row_data, columns=['filename', 'clip_idx', "prob"])
    results = results.sort_values(by=["filename", 'clip_idx']).reset_index()
    unique_names = results["filename"].unique()
    results = results.groupby('filename')
    vmae_16x4 = dict()
    # save inference results to pickle for post-processing
    for file_name in unique_names:
        vmae_16x4[file_name] = np.asarray(results.get_group(str(file_name))["prob"])
    output_dir = 'pickles_X3D/A2'
    if not os.path.exists(output_dir):
        os.makedirs('pickles_X3D/A2')
    with open(os.path.join(output_dir, "A2_{}_vmae_16x4_crop_fold{}.pkl".format(view, fold)), "wb") as f:
        pickle.dump(vmae_16x4, f)

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    seed_everything(0)

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
    checkpoint_dashboard_list = natsorted(checkpoint_dashboard_list)
    checkpoint_rearview_list = natsorted(checkpoint_rearview_list)
    checkpoint_right_list = natsorted(checkpoint_right_list)
    for i in range(len(checkpoint_dashboard_list)):
        print('Load checkpoint:', checkpoint_dashboard_list[i])
        main(cfg, 'A2/', 'video_ids.csv', cfg.DATA.SAMPLING_RATE, 30, True, 'dash', i, checkpoint_dashboard_list[i])
    for i in range(len(checkpoint_rearview_list)):
        print('Load checkpoint:', checkpoint_rearview_list[i])
        main(cfg, 'A2/', 'video_ids.csv', cfg.DATA.SAMPLING_RATE, 30, True, 'rear', i, checkpoint_rearview_list[i])
    for i in range(len(checkpoint_right_list)):
        print('Load checkpoint:', checkpoint_right_list[i])
        main(cfg, 'A2/', 'video_ids.csv', cfg.DATA.SAMPLING_RATE, 30, True, 'right', i, checkpoint_right_list[i])
