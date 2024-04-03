import pickle
import os
import warnings
import pandas as pd
import numpy as np
import re
from tools import util_loc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

_FILENAME_TO_ID = {'Dashboard_user_id_20090_NoAudio_5': 1, 'Dashboard_user_id_20090_NoAudio_7': 2, 'Dashboard_user_id_35418_NoAudio_5': 3, 'Dashboard_user_id_35418_NoAudio_7': 4,
                   'Dashboard_user_id_36305_NoAudio_3': 5, 'Dashboard_user_id_36305_NoAudio_5': 6, 'Dashboard_user_id_46856_NoAudio_5': 7, 'Dashboard_user_id_46856_NoAudio_7': 8, 'Dashboard_user_id_53307_NoAudio_5': 9,
                   'Dashboard_user_id_53307_NoAudio_7': 10, 'Dashboard_user_id_59014_NoAudio_5': 11, 'Dashboard_user_id_59014_NoAudio_7': 12, 'Dashboard_user_id_61302_NoAudio_7': 13, 'Dashboard_user_id_61302_NoAudio_9': 14,
                   'Dashboard_user_id_61497_NoAudio_5': 15, 'Dashboard_user_id_61497_NoAudio_7': 16, 'Dashboard_user_id_71436_NoAudio_5': 17, 'Dashboard_user_id_71436_NoAudio_7': 18, 'Dashboard_user_id_72446_NoAudio_5': 19,
                   'Dashboard_user_id_72446_NoAudio_7': 20, 'Dashboard_user_id_85870_NoAudio_5': 21, 'Dashboard_user_id_85870_NoAudio_7': 22, 'Dashboard_user_id_86356_NoAudio_5': 23, 'Dashboard_user_id_86356_NoAudio_7': 24,
                   'Dashboard_user_id_87903_NoAudio_5': 25, 'Dashboard_user_id_87903_NoAudio_7': 26}

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
    with open(os.path.join(pickle_dir, "A2_{}_vmae_16x4_crop_fold0.pkl".format(view)), "rb") as fp:
        vmae_16x4_probs = pickle.load(fp)
    probs.append(vmae_16x4_probs) # [5]
    return probs

def multi_view_ensemble(avg_dash_seq, avg_dash_seq_uniformer_full, avg_dash_seq_x3d):
    alpha, beta, sigma = 0.4, 0.3, 0.3
    prob_ensemble = alpha * avg_dash_seq + beta * avg_dash_seq_uniformer_full + sigma * avg_dash_seq_x3d
    return prob_ensemble

def main():
    clip_classification = []
    pickle_dir_videomae = "pickles_videomae_fold0/A2"
    k_flod_dash_probs = load_k_fold_probs(pickle_dir_videomae, "dash")

    pickle_dir_uniformer_full = "pickles_uniformer_fold0/A2"
    k_flod_dash_probs_uniformer_full = load_k_fold_probs(pickle_dir_uniformer_full, "dash")

    pickle_dir_x3d = "pickles_X3D_fold0/A2"
    k_flod_dash_probs_x3d = load_k_fold_probs(pickle_dir_x3d, "dash")

    for right_vid in k_flod_dash_probs[0].keys():
        dash_vid = "Dashboard_" + re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]
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
    with open("pred_dash_fold0.txt", "w") as fp:
        for (vid, label, start, end) in loc_segments:
            fp.writelines("{} {} {} {}\n".format(int(vid), label, start, end))
    df = pd.read_csv('pred_dash_fold0.txt', delimiter=' ', names=["ID", "Label (Primary)", "Start Time", "End Time"])
    df = df.sort_values(by=['ID', 'Start Time'])
    df.to_csv('pred_dash_fold0.txt', index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()