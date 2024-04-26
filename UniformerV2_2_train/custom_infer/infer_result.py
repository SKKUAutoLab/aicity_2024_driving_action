import os
import torch
import random
import cv2
import pandas as pd
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import numpy as np
import glob
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

def generate_gaussian_weights(sigma, length): # 30 and 32
    center = length // 2
    x = np.linspace(-center, center, length)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel

def generate_gaussian_weights_with_stride(data, length, stride, sigma=30):
    weights_list = []
    for i in range(0, len(data), stride):
        sub_data = data[i:i + length]
        if len(sub_data) == length:
            weights = generate_gaussian_weights(sigma, length)
            weights_list.append(weights)
    return np.concatenate(weights_list, axis=0) # [14592]

@torch.no_grad()
def ensemble_results(cfg, videoids):
    total_prob_sq = {}
    for key, values in videoids.items():
        video_path = values[1]
        print('Processing video:', video_path)
        # ensemble 36 models: 3*3*2*2
        overlap_ratio = [0, 0.25, 0.5, 0.75]
        # sample_rate1 = [4, 8, 12]
        # sample_rate2 = [2, 4, 6]
        # frame_nums = [8, 16]
        sample_rate = cfg.DATA.SAMPLING_RATE
        frame_nums = cfg.DATA.NUM_FRAMES
        use_gussion = True
        views = ['dashboard', 'rearview', 'right']
        # datatypes = ['original', 'expand']
        zeros_pad = [np.zeros((1, 16))]
        npyloader = []
        gussion_weights = []
        dashboard_result = []
        rearview_result = []
        right_result = []

        # for view in views:
        #     for frame_num in frame_nums:
        #         if frame_num == 16:
        #             sample_rate = sample_rate2
        #         else:
        #             sample_rate = sample_rate1
        #         for datatype in datatypes:
        #             for rate in sample_rate:
        #                 for ratio in overlap_ratio:
        #                     prob_file = 'probability_results/view-{}_frame-{}_rate-{}_datatype-{}_overlapratio-{}.npy'.format(view, frame_num, rate, datatype, ratio)
        #                     load_data = np.load(prob_file, allow_pickle=True).item()[str(values[0])] # [14592]
        #                     if ratio != 0:
        #                         zeros_pad_gussion = np.zeros(int(frame_num * rate * ratio), dtype=float)
        #                         npyloader.append(zeros_pad * int(frame_num * rate * ratio) + load_data) # [48]
        #                         gussion_weights.append(np.concatenate((zeros_pad_gussion, generate_gaussian_weights_with_stride(load_data, length=cfg.DATA.NUM_FRAMES * rate, stride=cfg.DATA.NUM_FRAMES * rate)))) # [48]
        #                     else:
        #                         npyloader.append(load_data) # [45]
        #                         gussion_weights.append(generate_gaussian_weights_with_stride(load_data, length=cfg.DATA.NUM_FRAMES * rate, stride=cfg.DATA.NUM_FRAMES * rate)) # [45]
        #     max_len = len(max(npyloader, key=len)) # 14664
        #     for sq in npyloader:
        #         sq.extend(zeros_pad * (max_len - len(sq))) # [14664]
        #     gussion_weights_new = []
        #     for sq in gussion_weights:
        #         gussion_weights_new.append(np.concatenate((sq, np.zeros(max_len - len(sq))))) # [48]
        #
        #     if use_gussion: # ensemble by gaussian
        #         count_final = []
        #         for pred, gussion_weight in zip(npyloader, gussion_weights_new):
        #             pred = np.array(pred)[:, 0] # [14664, 16]
        #             gussion_weight = gussion_weight[:, None] # [48]
        #             count_final.append(pred * gussion_weight)
        #         count_final = np.array(count_final) # [48, 14664, 16]
        #         gussion_weights_new = np.array(gussion_weights_new)[:, :, None] # [48, 14664, 1]
        #         final_result = (np.sum(count_final, axis=0) / np.sum(gussion_weights_new, axis=0))[:, None, :] # [14664, 1, 16]
        #     else: # ensemble by mean
        #         mask = np.ma.masked_where(npyloader == zeros_pad, npyloader)
        #         final_result = np.ma.mean(mask, axis=0)
        #     if view == 'dashboard':
        #         dashboard_result = final_result # [14664]
        #     elif view == 'rearview':
        #         rearview_result = final_result # [14664]
        #     elif view == 'right':
        #         right_result = final_result # [14664]
        #     npyloader = []
        #     gussion_weights = []

        for view in views:
            for ratio in overlap_ratio:
                prob_file = 'probability_results/view_{}_frame_{}_rate_{}_overlapratio_{}.npy'.format(view, frame_nums, sample_rate, ratio)
                load_data = np.load(prob_file, allow_pickle=True).item()[values[0]] # [14592]
                if ratio != 0:
                    zeros_pad_gussion = np.zeros(int(frame_nums * sample_rate * ratio), dtype=float)
                    npyloader.append(zeros_pad * int(frame_nums * sample_rate * ratio) + load_data) # [48]
                    gussion_weights.append(np.concatenate((zeros_pad_gussion, generate_gaussian_weights_with_stride(load_data, length=cfg.DATA.NUM_FRAMES * sample_rate, stride=cfg.DATA.NUM_FRAMES * sample_rate)))) # [48]
                else:
                    npyloader.append(load_data) # [45]
                    gussion_weights.append(generate_gaussian_weights_with_stride(load_data, length=cfg.DATA.NUM_FRAMES * sample_rate, stride=cfg.DATA.NUM_FRAMES * sample_rate)) # [45]
            max_len = len(max(npyloader, key=len)) # 14664
            for sq in npyloader:
                sq.extend(zeros_pad * (max_len - len(sq))) # [14664]
            gussion_weights_new = []
            for sq in gussion_weights:
                gussion_weights_new.append(np.concatenate((sq, np.zeros(max_len - len(sq))))) # [48]

            if use_gussion: # ensemble by gaussian
                count_final = []
                for pred, gussion_weight in zip(npyloader, gussion_weights_new):
                    pred = np.array(pred)[:, 0] # [14664, 16]
                    gussion_weight = gussion_weight[:, None] # [48]
                    count_final.append(pred * gussion_weight)
                count_final = np.array(count_final) # [48, 14664, 16]
                gussion_weights_new = np.array(gussion_weights_new)[:, :, None] # [48, 14664, 1]
                final_result = (np.sum(count_final, axis=0) / np.sum(gussion_weights_new, axis=0))[:, None, :] # [14664, 1, 16]
            else: # ensemble by mean
                mask = np.ma.masked_where(npyloader == zeros_pad, npyloader)
                final_result = np.ma.mean(mask, axis=0)
            if view == 'dashboard':
                dashboard_result = final_result # [14664]
            elif view == 'rearview':
                rearview_result = final_result # [14664]
            elif view == 'right':
                right_result = final_result # [14664]
            npyloader = []
            gussion_weights = []

        dashboard_result = dashboard_result.tolist()
        rearview_result = rearview_result.tolist()
        right_result = right_result.tolist()
        max_len = len(max([dashboard_result, rearview_result, right_result], key=len)) # 14664
        for sq in [dashboard_result, rearview_result, right_result]:
            sq.extend(zeros_pad * (max_len - len(sq))) # [14664]
        dashboard_result = np.array(dashboard_result) # [14664, 1, 16]
        rearview_result = np.array(rearview_result) # [14664, 1, 16]
        right_result = np.array(right_result) # [14664, 1, 16]
        dashboard_weight = np.ones((1, 16))
        rearview_weight = np.ones((1, 16))
        right_weight = np.ones((1, 16))
        dashboard_class = [3, 4, 13, 15]
        rearview_class = [3, 4, 13, 15]
        rightview_class = [5, 8, 11, 15]
        for i in dashboard_class:
            dashboard_weight[0][i] = 1000
        for i in rearview_class:
            rearview_weight[0][i] = 1000
        for i in rightview_class:
            right_weight[0][i] = 1000
            if i == 15:
                right_weight[0][i] = 500

        # weight normalization
        sum_weight = dashboard_weight + rearview_weight + right_weight
        dashboard_weight /= sum_weight
        rearview_weight /= sum_weight
        right_weight /= sum_weight

        # weighted summation
        dashboard_result *= dashboard_weight
        rearview_result *= rearview_weight
        right_result *= right_weight
        fusion_result = dashboard_result + rearview_result + right_result
        total_prob_sq[values[0]] = fusion_result
    return dict(sorted(total_prob_sq.items()))

def get_classification(sequence_class_prob):
    labels_index = np.argmax(sequence_class_prob, axis=1) # [14664], returns list of position of max value in each list
    probs = np.max(sequence_class_prob, axis=1)  # [14664], return list of max value in each  list.
    return labels_index, probs

def activity_localization(prob_sq):
    action_idx, action_probs = get_classification(prob_sq)
    ts = 0.5
    class_confidence = {i: [0, 0] for i in range(0, 16)}
    for i in range(len(action_idx)):
        class_confidence[action_idx[i]][0] += action_probs[i]
        class_confidence[action_idx[i]][1] += 1
    result = []
    for key in sorted(class_confidence.keys()):
        if class_confidence[key][1] > 0:
            result.append(class_confidence[key][0] / class_confidence[key][1])
        else:
            result.append(ts)
    result = [i - 0.1 for i in result] # [16]
    result = [i if i > 0.1 else 0.1 for i in result]
    threshold = [ts if x > ts else x for x in result]
    action_tag = np.array([1 if action_probs[i] > threshold[action_idx[i]] else 0 for i in range(len(action_probs))])
    activities_idx = []
    startings = []
    endings = []
    action_probility = []
    for i in range(len(action_tag)):
        if action_tag[i] == 1:
            activities_idx.append(action_idx[i])
            action_probility.append(action_probs[i])
            start = i
            end = i + 1
            startings.append(start)
            endings.append(end)
    return activities_idx, startings, endings, action_probility

def merge_and_remove(data):
    df_total = pd.DataFrame([[0, 0, 0, 0, 0]], columns=[0, 1, 2, 3, 4])
    for i in range(1, 31):
        data_video = data[data[0] == i]
        list_label = data_video[1].unique()
        for label in list_label:
            data_video_label = data_video[data_video[1] == label]
            data_video_label = data_video_label.reset_index()
            countrow = 1
            for j in range(len(data_video_label) - 1):
                # start2 - end1 <= 8
                if data_video_label.loc[j + 1, 2] - data_video_label.loc[j, 3] <= 8: # merge two clips with same class
                    data_video_label.loc[j + 1, 2] = data_video_label.loc[j, 2] # start2 = start1
                    data_video_label.loc[j, 3] = 0 # end1 = 0
                    data_video_label.loc[j, 2] = 0 # start1 = 0
            for j in range(len(data_video_label)):
                # start1 = 0 and end1 = 0
                if data_video_label.loc[j, 2] == 0 and data_video_label.loc[j, 3] == 0:
                    countrow += 1
                    data_video_label.loc[j + 1, 4] += data_video_label.loc[j, 4] # prob2 += prob1
                    data_video_label.loc[j, 4] = 0 # prob1 = 0
                else:
                    data_video_label.loc[j, 4] = data_video_label.loc[j, 4] / countrow # prob1 = prob1 / countrow
                    countrow = 1
            df_total = df_total.append(data_video_label)
    df_total = df_total[df_total[3] != 0] # end_time != 0
    df_total = df_total[1 < df_total[3] - df_total[2]] # end_time - start_time > 1
    df_total = df_total[df_total[3] - df_total[2] < 30] # end_time - start_time < 30
    df_total = df_total.sort_values(by=[0, 2])

    drop_index = []
    for num in range(len(df_total) - 1):
        row = df_total.iloc[num]
        next_row = df_total.iloc[num + 1]
        if next_row[2] - row[3] <= 5: # start2 - end1 <= 5
            if (row[1] == 2 or row[1] == 3) and (next_row[1] == 5 or next_row[1] == 6): # phone call first, then texting
                drop_index.append(df_total.iloc[num + 1]['index'])
            elif (row[1] == 5 or row[1] == 6) and (next_row[1] == 2 or next_row[1] == 3): # texting first, then phone call
                drop_index.append(df_total.iloc[num]['index'])
            if (row[1] == 14 and next_row[1] == 3): # hand on head first, then drinking
                df_total.iloc[num, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2]) # new start_time
                df_total.iloc[num, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3]) # new end_time
                drop_index.append(df_total.iloc[num + 1]['index'])
            elif (next_row[1] == 14 and row[1] == 3): # phone call (left) first, then hand on head
                df_total.iloc[num + 1, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2]) # new start_time
                df_total.iloc[num + 1, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3]) # new end_time
                drop_index.append(df_total.iloc[num]['index'])
            if (row[1] == 11 and next_row[1] == 12): # talk passenger right first, then talk passenger backseat
                df_total.iloc[num, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2]) # new start_time
                df_total.iloc[num, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3]) # new end_time
                drop_index.append(df_total.iloc[num + 1]['index'])
            elif (next_row[1] == 11 and row[1] == 12): # talk passenger backseat first, then talk passenger right
                df_total.iloc[num + 1, 2] = min(df_total.iloc[num, 2], df_total.iloc[num + 1, 2]) # new start_time
                df_total.iloc[num + 1, 3] = max(df_total.iloc[num, 3], df_total.iloc[num + 1, 3]) # new end_time
                drop_index.append(df_total.iloc[num]['index'])
    drop_index = list(set(drop_index)) # [2]
    for d_index in drop_index:
        df_total = df_total[df_total['index'] != d_index]

    final_result = pd.DataFrame([[0, 0, 0, 0, 0]], columns=[0, 1, 2, 3, 4])
    for i in range(1, 31):
        data_video = df_total[df_total[0] == i]
        list_label = data_video[1].unique()
        for label in list_label:
            data_video_label = data_video[data_video[1] == label]
            data_video_label = data_video_label.reset_index()
            if len(data_video_label) == 0:
                continue
            elif len(data_video_label) == 1:
                append_item = data_video_label.iloc[0]
            else:
                maxprob_index = data_video_label[4].idxmax() # 0
                second_maxprob_index = data_video_label[4].drop(maxprob_index).idxmax() # 1, 2
                if data_video_label.loc[second_maxprob_index][4] > 0.95 * data_video_label.loc[maxprob_index][4]:
                    append_item_second = data_video_label.loc[second_maxprob_index]
                    final_result = final_result.append(append_item_second)
                append_item = data_video_label.loc[maxprob_index]
            data_video_label = append_item
            final_result = final_result.append(data_video_label)
    final_result = final_result.drop(4, axis=1)
    final_result = final_result.iloc[1:]
    final_result = final_result.drop(columns=['level_0', 'index'])
    final_result = final_result.astype('int').sort_values(by=[0, 2])
    df_total = df_total.drop(columns=['index'])
    df_total = final_result
    df_total.to_csv('final_submission.txt', sep=' ', index=False, header=False)

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
    labels = list(np.arange(16)) # 18 classes for dataset 2021
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
    prob_ensemble = ensemble_results(cfg, vid_info)

    # post-processing
    dataframe_list = []
    for i in range(1, len(vid_info) + 1): # 11
        len_prob = len(prob_ensemble[i]) # 14664
        prob_ensemble_video = []
        for ids in range(len_prob):
            prob_sub_mean = prob_ensemble[i][ids]
            prob_ensemble_video.append(prob_sub_mean)
        # classification step
        prob_actions = np.array(prob_ensemble_video) # [14664, 1, 16]
        prob_actions = np.squeeze(prob_actions) # [14664, 16]
        # temporal localization step
        activities_idx, startings, endings, activities_probility = activity_localization(prob_actions)
        for idx, s, e, p in zip(activities_idx, startings, endings, activities_probility):
            start = s / fps
            end = e / fps
            label = labels[idx]
            dataframe_list.append([i, label, start, end, p])
    data = pd.DataFrame(dataframe_list, columns=[0, 1, 2, 3, 4])
    general_submission(data)
