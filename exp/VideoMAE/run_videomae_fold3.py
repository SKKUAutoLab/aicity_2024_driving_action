import pickle
import os
import warnings
import pandas as pd
import numpy as np
import re
from tools import util_loc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

_FILENAME_TO_ID = {'Rear_view_user_id_47158_NoAudio_5': 1, 'Rear_view_user_id_47158_NoAudio_7': 2, 'Rear_view_user_id_47457_NoAudio_5': 3, 'Rear_view_user_id_47457_NoAudio_7': 4, 'Rear_view_user_id_52046_NoAudio_11': 5,
                   'Rear_view_user_id_52046_NoAudio_5': 6, 'Rear_view_user_id_64003_NoAudio_5': 7, 'Rear_view_user_id_64003_NoAudio_7': 8, 'Rear_view_user_id_93491_NoAudio_5': 9, 'Rear_view_user_id_93491_NoAudio_7': 10,
                   'Dashboard_user_id_47158_NoAudio_5': 1, 'Dashboard_user_id_47158_NoAudio_7': 2, 'Dashboard_user_id_47457_NoAudio_5': 3, 'Dashboard_user_id_47457_NoAudio_7': 4, 'Dashboard_user_id_52046_NoAudio_11': 5,
                   'Dashboard_user_id_52046_NoAudio_5': 6, 'Dashboard_user_id_64003_NoAudio_5': 7, 'Dashboard_user_id_64003_NoAudio_7': 8, 'Dashboard_user_id_93491_NoAudio_5': 9, 'Dashboard_user_id_93491_NoAudio_7': 10,
                   'Right_side_window_user_id_47158_NoAudio_5': 1, 'Right_side_window_user_id_47158_NoAudio_7': 2, 'Right_side_window_user_id_47457_NoAudio_5': 3, 'Right_side_window_user_id_47457_NoAudio_7': 4,
                   'Right_side_window_user_id_52046_NoAudio_11': 5, 'Right_side_window_user_id_52046_NoAudio_5': 6, 'Right_side_window_user_id_64003_NoAudio_5': 7, 'Right_side_window_user_id_64003_NoAudio_7': 8,
                   'Right_side_window_user_id_93491_NoAudio_5': 9, 'Right_side_window_user_id_93491_NoAudio_7': 10}

def get_classification(sequence_class_prob):
    labels_index = np.argmax(sequence_class_prob, axis=1)
    probs = np.max(sequence_class_prob, axis=1)
    return labels_index, probs

def activity_localization(prob_sq, vid, action_threshold):
    action_idx, action_probs = get_classification(prob_sq)
    action_tag = np.zeros(action_idx.shape)
    action_tag[action_probs >= action_threshold] = 1
    activities_idx = []
    startings = []
    endings = []
    clip_classfication = []
    for i in range(len(action_tag)):
        if action_tag[i] == 1:
            activities_idx.append(action_idx[i])
            start = i
            end = i + 1
            startings.append(start)
            endings.append(end)
            clip_classfication.append([int(vid), action_idx[i], start, end])
    return clip_classfication

def smoothing(x, k=3):
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        if int(np.argmax(x[i], axis=-1)) == 14:
            y[i] = x[i]
        else:
            y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

def load_k_fold_probs(pickle_dir, view):
    probs = []
    with open(os.path.join(pickle_dir, "A2_{}_vmae_16x4_crop_fold3.pkl").format(view), "rb") as fp:
        vmae_16x4_probs = pickle.load(fp)
    probs.append(vmae_16x4_probs)
    return probs

def multi_view_ensemble(avg_dash_seq_videomae, avg_right_seq_videomae, avg_rear_seq_videomae):
    alpha, beta, sigma = 0.3, 0.4, 0.3
    prob_ensemble = alpha * avg_dash_seq_videomae + beta * avg_right_seq_videomae + sigma * avg_rear_seq_videomae
    prob_ensemble[:, 3:4] = np.array(avg_rear_seq_videomae[:, 3:4])
    prob_ensemble[:, 4:5] = prob_ensemble[:, 4:5]
    prob_ensemble[:, 5:6] = np.array(avg_right_seq_videomae)[:, 5:6]
    prob_ensemble[:, 6:7] = np.array(avg_right_seq_videomae)[:, 6:7]
    prob_ensemble[:, 7:9] = np.array(avg_right_seq_videomae)[:, 7:9]
    prob_ensemble[:, 13:14] = (np.array(avg_dash_seq_videomae)[:, 13:14]) * 2
    prob_ensemble[:, 14:15] = np.array(avg_rear_seq_videomae)[:, 14:15] * 1.5
    prob_ensemble[:, 15:] = np.array(avg_right_seq_videomae)[:, 15:] * 1.5
    return prob_ensemble

def main():
    clip_classification = []
    pickle_dir_videomae = "pickles_videomae_fold3/A2"
    k_flod_dash_probs_videomae = load_k_fold_probs(pickle_dir_videomae, "dash")
    k_flod_right_probs_videomae = load_k_fold_probs(pickle_dir_videomae, "right")
    k_flod_rear_probs_videomae = load_k_fold_probs(pickle_dir_videomae, "rear")
    all_model_results = dict()

    for right_vid in k_flod_right_probs_videomae[0].keys():
        dash_vid = "Dashboard_" + re.search("user_id_[0-9]{5}_NoAudio_[0-9]{1,2}", right_vid)[0]
        rear_vid = "Rear_view_" + re.search("user_id_[0-9]{5}_NoAudio_[0-9]{1,2}", right_vid)[0]
        all_dash_probs_videomae = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs_videomae])
        all_right_probs_videomae = np.stack([np.array(list(map(np.array, right_prob[right_vid]))) for right_prob in k_flod_right_probs_videomae])
        all_rear_probs_videomae = np.stack([np.array(list(map(np.array, rear_prob[rear_vid]))) for rear_prob in k_flod_rear_probs_videomae])

        avg_dash_seq_videomae = np.mean(all_dash_probs_videomae, axis=0)
        avg_right_seq_videomae = np.mean(all_right_probs_videomae, axis=0)
        avg_rear_seq_videomae = np.mean(all_rear_probs_videomae, axis=0)

        avg_dash_seq_videomae = smoothing(np.array(avg_dash_seq_videomae), k=1)
        avg_right_seq_videomae = smoothing(np.array(avg_right_seq_videomae), k=1)
        avg_rear_seq_videomae = smoothing(np.array(avg_rear_seq_videomae), k=1)

        length_videomae = min(avg_dash_seq_videomae.shape[0], avg_right_seq_videomae.shape[0], avg_rear_seq_videomae.shape[0])
        avg_dash_seq_videomae, avg_right_seq_videomae, avg_rear_seq_videomae = avg_dash_seq_videomae[:length_videomae, :], avg_right_seq_videomae[:length_videomae, :], avg_rear_seq_videomae[:length_videomae, :]

        prob_ensemble = multi_view_ensemble(avg_dash_seq_videomae, avg_right_seq_videomae, avg_rear_seq_videomae)
        vid = _FILENAME_TO_ID[right_vid]
        prob_seq = np.array(prob_ensemble)
        prob_seq = np.squeeze(prob_seq)
        clip_classification += activity_localization(prob_seq, vid, 0.1)
        all_model_results[vid] = {"dash": all_dash_probs_videomae, "rear": all_rear_probs_videomae, "right": all_right_probs_videomae}

    clip_classification = pd.DataFrame(clip_classification, columns=["video_id", "label", "start", "end"])
    loc_segments = util_loc.clip_to_segment(clip_classification)
    loc_segments = util_loc.reclassify_segment(loc_segments, all_model_results)
    loc_segments = util_loc.correct_with_prior_constraints(loc_segments)
    with open("pred_videomae_fold3.txt", "w") as fp:
        for (vid, label, start, end) in loc_segments:
            fp.writelines("{} {} {} {}\n".format(int(vid), label, start, end))
    df = pd.read_csv('pred_videomae_fold3.txt', delimiter=' ', names=["ID", "Label (Primary)", "Start Time", "End Time"])
    df = df.sort_values(by=['ID', 'Start Time'])
    df.to_csv('pred_videomae_fold3.txt', index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()