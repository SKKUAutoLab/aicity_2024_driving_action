import pickle
import os
import warnings
import pandas as pd
import numpy as np
import re
from tools import util_loc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

_FILENAME_TO_ID = {'Rear_view_user_id_13522_NoAudio_5': 1, 'Rear_view_user_id_13522_NoAudio_7': 2, 'Rear_view_user_id_14786_NoAudio_5': 3, 'Rear_view_user_id_14786_NoAudio_7': 4, 'Rear_view_user_id_16080_NoAudio_5': 5,
                   'Rear_view_user_id_16080_NoAudio_7': 6, 'Rear_view_user_id_20507_NoAudio_5': 7, 'Rear_view_user_id_20507_NoAudio_7': 8, 'Rear_view_user_id_22640_NoAudio_5': 9, 'Rear_view_user_id_22640_NoAudio_7': 10,
                   'Rear_view_user_id_25432_NoAudio_5': 11, 'Rear_view_user_id_25432_NoAudio_7': 12, 'Rear_view_user_id_28557_NoAudio_5': 13, 'Rear_view_user_id_28557_NoAudio_7': 14, 'Rear_view_user_id_30932_NoAudio_5': 15,
                   'Rear_view_user_id_30932_NoAudio_7': 16, 'Rear_view_user_id_31903_NoAudio_5': 17, 'Rear_view_user_id_31903_NoAudio_7': 18, 'Rear_view_user_id_33682_NoAudio_5': 19, 'Rear_view_user_id_33682_NoAudio_7': 20,
                   'Rear_view_user_id_38159_NoAudio_5': 21, 'Rear_view_user_id_38159_NoAudio_7': 22, 'Rear_view_user_id_38479_NoAudio_5': 23, 'Rear_view_user_id_38479_NoAudio_7': 24, 'Rear_view_user_id_39367_NoAudio_5': 25,
                   'Rear_view_user_id_39367_NoAudio_7': 26, 'Rear_view_user_id_40908_NoAudio_5': 27, 'Rear_view_user_id_40908_NoAudio_7': 28, 'Rear_view_user_id_42711_NoAudio_5': 29, 'Rear_view_user_id_42711_NoAudio_7': 30,
                   'Rear_view_user_id_46141_NoAudio_5': 31, 'Rear_view_user_id_46141_NoAudio_7': 32, 'Rear_view_user_id_46844_NoAudio_5': 33, 'Rear_view_user_id_46844_NoAudio_7': 34, 'Rear_view_user_id_50347_NoAudio_5': 35,
                   'Rear_view_user_id_50347_NoAudio_7': 36, 'Rear_view_user_id_50921_NoAudio_5': 37, 'Rear_view_user_id_50921_NoAudio_7': 38, 'Rear_view_user_id_59581_NoAudio_5': 39, 'Rear_view_user_id_59581_NoAudio_7': 40,
                   'Rear_view_user_id_60167_NoAudio_5': 41, 'Rear_view_user_id_60167_NoAudio_7': 42, 'Rear_view_user_id_60768_NoAudio_5': 43, 'Rear_view_user_id_60768_NoAudio_7': 44, 'Rear_view_user_id_63513_NoAudio_5': 45,
                   'Rear_view_user_id_63513_NoAudio_7': 46, 'Rear_view_user_id_63764_NoAudio_5': 47, 'Rear_view_user_id_63764_NoAudio_7': 48, 'Rear_view_user_id_69039_NoAudio_5': 49, 'Rear_view_user_id_69039_NoAudio_7': 50,
                   'Rear_view_user_id_70176_NoAudio_5': 51, 'Rear_view_user_id_70176_NoAudio_7': 52, 'Rear_view_user_id_71720_NoAudio_5': 53, 'Rear_view_user_id_71720_NoAudio_7': 54, 'Rear_view_user_id_79023_NoAudio_5': 55,
                   'Rear_view_user_id_79023_NoAudio_7': 56, 'Rear_view_user_id_79755_NoAudio_3': 57, 'Rear_view_user_id_79755_NoAudio_5': 58, 'Rear_view_user_id_83323_NoAudio_3': 59, 'Rear_view_user_id_83323_NoAudio_5': 60,
                   'Rear_view_user_id_83756_NoAudio_3': 61, 'Rear_view_user_id_83756_NoAudio_7': 62, 'Rear_view_user_id_84935_NoAudio_5': 63, 'Rear_view_user_id_84935_NoAudio_7': 64, 'Rear_view_user_id_86952_NoAudio_11': 65,
                   'Rear_view_user_id_86952_NoAudio_9': 66, 'Rear_view_user_id_93439_NoAudio_5': 67, 'Rear_view_user_id_93439_NoAudio_7': 68, 'Rear_view_user_id_93542_NoAudio_5': 69, 'Rear_view_user_id_93542_NoAudio_7': 70,
                   'Rear_view_user_id_96371_NoAudio_7': 71, 'Rear_view_user_id_96371_NoAudio_9': 72, 'Rear_view_user_id_98389_NoAudio_5': 73, 'Rear_view_user_id_98389_NoAudio_7': 74, 'Rear_view_user_id_99660_NoAudio_5': 75,
                   'Rear_view_user_id_99660_NoAudio_7': 76, 'Rear_view_user_id_99882_NoAudio_5': 77, 'Rear_view_user_id_99882_NoAudio_7': 78}

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
    with open(os.path.join(pickle_dir, "A2_{}_vmae_16x4_crop_fold4.pkl".format(view)), "rb") as fp:
        vmae_16x4_probs = pickle.load(fp)
    probs.append(vmae_16x4_probs) # [5]
    return probs

def multi_view_ensemble(avg_dash_seq, avg_dash_seq_uniformer_full, avg_dash_seq_x3d):
    alpha, beta, sigma = 0.4, 0.3, 0.3
    prob_ensemble = alpha * avg_dash_seq + beta * avg_dash_seq_uniformer_full + sigma * avg_dash_seq_x3d
    return prob_ensemble

def main():
    clip_classification = []
    pickle_dir_videomae = "pickles_videomae_fold4/A2"
    k_flod_dash_probs = load_k_fold_probs(pickle_dir_videomae, "rear")

    pickle_dir_uniformer_full = "pickles_uniformer_fold4/A2"
    k_flod_dash_probs_uniformer_full = load_k_fold_probs(pickle_dir_uniformer_full, "rear")

    pickle_dir_x3d = "pickles_X3D_fold4/A2"
    k_flod_dash_probs_x3d = load_k_fold_probs(pickle_dir_x3d, "rear")

    for right_vid in k_flod_dash_probs[0].keys():
        dash_vid = "Rear_view_" + re.search("user_id_[0-9]{5}_NoAudio_[0-9]{1,2}", right_vid)[0]
        all_dash_probs = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs])
        all_dash_probs_uniformer_full = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs_uniformer_full])
        all_dash_probs_x3d = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs_x3d])

        avg_dash_seq = np.mean(all_dash_probs, axis=0)
        avg_dash_seq_uniformer_full = np.mean(all_dash_probs_uniformer_full, axis=0)
        avg_dash_seq_x3d = np.mean(all_dash_probs_x3d, axis=0)

        avg_dash_seq = smoothing(np.array(avg_dash_seq), k=1)
        avg_dash_seq_uniformer_full = smoothing(np.array(avg_dash_seq_uniformer_full), k=1)
        avg_dash_seq_x3d = smoothing(np.array(avg_dash_seq_x3d), k=1)

        prob_ensemble = multi_view_ensemble(avg_dash_seq, avg_dash_seq_uniformer_full, avg_dash_seq_x3d)
        vid = _FILENAME_TO_ID[right_vid]
        prob_seq = np.array(prob_ensemble)
        prob_seq = np.squeeze(prob_seq)
        clip_classification += activity_localization(prob_seq, vid, 0.1) # default 0.1

    clip_classification = pd.DataFrame(clip_classification, columns=["video_id", "label", "start", "end"])
    loc_segments = util_loc.clip_to_segment(clip_classification)
    loc_segments = util_loc.correct_with_prior_constraints(loc_segments)
    with open("pred_rear_fold4.txt", "w") as fp:
        for (vid, label, start, end) in loc_segments:
            fp.writelines("{} {} {} {}\n".format(int(vid), label, start, end))
    df = pd.read_csv('pred_rear_fold4.txt', delimiter=' ', names=["ID", "Label (Primary)", "Start Time", "End Time"])
    df = df.sort_values(by=['ID', 'Start Time'])
    df.to_csv('pred_rear_fold4.txt', index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()