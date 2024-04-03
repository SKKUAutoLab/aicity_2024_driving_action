import pickle
import os
import warnings
import pandas as pd
import numpy as np
import re
from tools import util_loc
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

_FILENAME_TO_ID = {'Rear_view_user_id_20090_NoAudio_5': 1, 'Rear_view_user_id_20090_NoAudio_7': 2, 'Rear_view_user_id_35418_NoAudio_5': 3, 'Rear_view_user_id_35418_NoAudio_7': 4, 'Rear_view_user_id_36305_NoAudio_3': 5,
                   'Rear_view_user_id_36305_NoAudio_5': 6, 'Rear_view_user_id_46856_NoAudio_5': 7, 'Rear_view_user_id_46856_NoAudio_7': 8, 'Rear_view_user_id_53307_NoAudio_5': 9, 'Rear_view_user_id_53307_NoAudio_7': 10,
                   'Rear_view_user_id_59014_NoAudio_5': 11, 'Rear_view_user_id_59014_NoAudio_7': 12, 'Rear_view_user_id_61302_NoAudio_7': 13, 'Rear_view_user_id_61302_NoAudio_9': 14, 'Rear_view_user_id_61497_NoAudio_5': 15,
                   'Rear_view_user_id_61497_NoAudio_7': 16, 'Rear_view_user_id_71436_NoAudio_5': 17, 'Rear_view_user_id_71436_NoAudio_7': 18, 'Rear_view_user_id_72446_NoAudio_5': 19, 'Rear_view_user_id_72446_NoAudio_7': 20,
                   'Rear_view_user_id_85870_NoAudio_5': 21, 'Rear_view_user_id_85870_NoAudio_7': 22, 'Rear_view_user_id_86356_NoAudio_5': 23, 'Rear_view_user_id_86356_NoAudio_7': 24, 'Rear_view_user_id_87903_NoAudio_5': 25,
                   'Rear_view_user_id_87903_NoAudio_7': 26, 'Dashboard_user_id_20090_NoAudio_5': 1, 'Dashboard_user_id_20090_NoAudio_7': 2, 'Dashboard_user_id_35418_NoAudio_5': 3, 'Dashboard_user_id_35418_NoAudio_7': 4,
                   'Dashboard_user_id_36305_NoAudio_3': 5, 'Dashboard_user_id_36305_NoAudio_5': 6, 'Dashboard_user_id_46856_NoAudio_5': 7, 'Dashboard_user_id_46856_NoAudio_7': 8, 'Dashboard_user_id_53307_NoAudio_5': 9,
                   'Dashboard_user_id_53307_NoAudio_7': 10, 'Dashboard_user_id_59014_NoAudio_5': 11, 'Dashboard_user_id_59014_NoAudio_7': 12, 'Dashboard_user_id_61302_NoAudio_7': 13, 'Dashboard_user_id_61302_NoAudio_9': 14,
                   'Dashboard_user_id_61497_NoAudio_5': 15, 'Dashboard_user_id_61497_NoAudio_7': 16, 'Dashboard_user_id_71436_NoAudio_5': 17, 'Dashboard_user_id_71436_NoAudio_7': 18, 'Dashboard_user_id_72446_NoAudio_5': 19,
                   'Dashboard_user_id_72446_NoAudio_7': 20, 'Dashboard_user_id_85870_NoAudio_5': 21, 'Dashboard_user_id_85870_NoAudio_7': 22, 'Dashboard_user_id_86356_NoAudio_5': 23, 'Dashboard_user_id_86356_NoAudio_7': 24,
                   'Dashboard_user_id_87903_NoAudio_5': 25, 'Dashboard_user_id_87903_NoAudio_7': 26, 'Right_side_window_user_id_20090_NoAudio_5': 1, 'Right_side_window_user_id_20090_NoAudio_7': 2,
                   'Right_side_window_user_id_35418_NoAudio_5': 3, 'Right_side_window_user_id_35418_NoAudio_7': 4, 'Right_side_window_user_id_36305_NoAudio_3': 5, 'Right_side_window_user_id_36305_NoAudio_5': 6,
                   'Right_side_window_user_id_46856_NoAudio_5': 7, 'Right_side_window_user_id_46856_NoAudio_7': 8, 'Right_side_window_user_id_53307_NoAudio_5': 9, 'Right_side_window_user_id_53307_NoAudio_7': 10,
                   'Right_side_window_user_id_59014_NoAudio_5': 11, 'Right_side_window_user_id_59014_NoAudio_7': 12, 'Right_side_window_user_id_61302_NoAudio_7': 13, 'Right_side_window_user_id_61302_NoAudio_9': 14,
                   'Right_side_window_user_id_61497_NoAudio_5': 15, 'Right_side_window_user_id_61497_NoAudio_7': 16, 'Right_side_window_user_id_71436_NoAudio_5': 17, 'Right_side_window_user_id_71436_NoAudio_7': 18,
                   'Right_side_window_user_id_72446_NoAudio_5': 19, 'Right_side_window_user_id_72446_NoAudio_7': 20, 'Right_side_window_user_id_85870_NoAudio_5': 21, 'Right_side_window_user_id_85870_NoAudio_7': 22,
                   'Right_side_window_user_id_86356_NoAudio_5': 23, 'Right_side_window_user_id_86356_NoAudio_7': 24, 'Right_side_window_user_id_87903_NoAudio_5': 25, 'Right_side_window_user_id_87903_NoAudio_7': 26}

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

def multi_view_ensemble(avg_dash_seq, avg_right_seq, avg_rear_seq, avg_dash_seq_uniformer_full, avg_right_seq_uniformer_full, avg_rear_seq_uniformer_full, avg_dash_seq_x3d, avg_right_seq_x3d, avg_rear_seq_x3d):
    alpha, beta, sigma = 0.3, 0.4, 0.3
    prob_ensemble = alpha * (avg_dash_seq + avg_dash_seq_uniformer_full) + beta * avg_right_seq + sigma * avg_rear_seq
    prob_ensemble[:, 1:2] = np.array(avg_rear_seq)[:, 1:2] + np.array(avg_dash_seq)[:, 1:2] * 0.4 # drinking
    prob_ensemble[:, 2:3] = (np.array(avg_right_seq)[:, 2:3] + np.array(avg_right_seq_x3d)[:, 2:3]) * 1.3 # phone call right
    prob_ensemble[:, 3:4] = (np.array(avg_rear_seq)[:, 3:4] + np.array(avg_rear_seq_x3d)[:, 3:4]) * 1.2 # phone call left
    prob_ensemble[:, 4:5] = prob_ensemble[:, 4:5] # eating
    prob_ensemble[:, 5:6] = (np.array(avg_right_seq)[:, 5:6] + np.array(avg_right_seq_x3d)[:, 5:6]) * 1.2 # texting right
    prob_ensemble[:, 6:7] = (np.array(avg_right_seq)[:, 6:7] + np.array(avg_right_seq_x3d)[:, 6:7]) * 1.2 # texting left
    prob_ensemble[:, 7:8] = (np.array(avg_right_seq)[:, 7:8] + np.array(avg_rear_seq[:, 7:8]) + np.array(avg_right_seq_x3d)[:, 7:8] + np.array(avg_rear_seq_x3d)[:, 7:8]) * 0.5 # reaching behind
    prob_ensemble[:, 8:9] = (np.array(avg_right_seq)[:, 8:9] + np.array(avg_rear_seq)[:, 8:9] + np.array(avg_rear_seq_x3d)[:, 8:9] + np.array(avg_right_seq_x3d)[:, 8:9]) * 0.5 # adjust control panel
    prob_ensemble[:, 9:10] = prob_ensemble[:, 9:10]  # picking up from floor driver
    prob_ensemble[:, 10:11] = prob_ensemble[:, 10:11] * 1.3  # picking up from floor passenger
    prob_ensemble[:, 11:12] = prob_ensemble[:, 11:12] * 0.9  # talking to passenger at right
    prob_ensemble[:, 12:13] = (np.array(avg_right_seq)[:, 2:3] + np.array(avg_right_seq_x3d)[:, 12:13]) * 1.1 # talking to passenger at backseat
    prob_ensemble[:, 13:14] = prob_ensemble[:, 13:14] * 3.9 # yawning
    prob_ensemble[:, 14:15] = (np.array(avg_rear_seq)[:, 14:15] + np.array(avg_rear_seq_x3d)[:, 14:15]) * 1.3 # hand on head
    prob_ensemble[:, 15:] = (np.array(avg_right_seq)[:, 15:] + np.array(avg_right_seq_x3d)[:, 15:]) * 0.9 # singing or dance with music
    return prob_ensemble

def main():
    clip_classification = []
    pickle_dir_videomae = "pickles_videomae_fold0/A2"
    k_flod_dash_probs = load_k_fold_probs(pickle_dir_videomae, "dash")
    k_flod_right_probs = load_k_fold_probs(pickle_dir_videomae, "right")
    k_flod_rear_probs = load_k_fold_probs(pickle_dir_videomae, "rear")

    pickle_dir_uniformer_full = "pickles_uniformer_fold0/A2"
    k_flod_dash_probs_uniformer_full = load_k_fold_probs(pickle_dir_uniformer_full, "dash")
    k_flod_right_probs_uniformer_full = load_k_fold_probs(pickle_dir_uniformer_full, "right")
    k_flod_rear_probs_uniformer_full = load_k_fold_probs(pickle_dir_uniformer_full, "rear")

    pickle_dir_x3d = "pickles_X3D_fold0/A2"
    k_flod_dash_probs_x3d = load_k_fold_probs(pickle_dir_x3d, "dash")
    k_flod_right_probs_x3d = load_k_fold_probs(pickle_dir_x3d, "right")
    k_flod_rear_probs_x3d = load_k_fold_probs(pickle_dir_x3d, "rear")

    all_model_results = dict()
    for right_vid in k_flod_right_probs[0].keys():
        dash_vid = "Dashboard_" + re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]
        rear_vid = "Rear_view_" + re.search("user_id_[0-9]{5}_NoAudio_[0-9]", right_vid)[0]

        all_dash_probs = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs])
        all_right_probs = np.stack([np.array(list(map(np.array, right_prob[right_vid]))) for right_prob in k_flod_right_probs])
        all_rear_probs = np.stack([np.array(list(map(np.array, rear_prob[rear_vid]))) for rear_prob in k_flod_rear_probs])

        all_dash_probs_uniformer_full = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs_uniformer_full])
        all_right_probs_uniformer_full = np.stack([np.array(list(map(np.array, right_prob[right_vid]))) for right_prob in k_flod_right_probs_uniformer_full])
        all_rear_probs_uniformer_full = np.stack([np.array(list(map(np.array, rear_prob[rear_vid]))) for rear_prob in k_flod_rear_probs_uniformer_full])

        all_dash_probs_x3d = np.stack([np.array(list(map(np.array, dash_prob[dash_vid]))) for dash_prob in k_flod_dash_probs_x3d])
        all_right_probs_x3d = np.stack([np.array(list(map(np.array, right_prob[right_vid]))) for right_prob in k_flod_right_probs_x3d])
        all_rear_probs_x3d = np.stack([np.array(list(map(np.array, rear_prob[rear_vid]))) for rear_prob in k_flod_rear_probs_x3d])

        avg_dash_seq = np.mean(all_dash_probs, axis=0)
        avg_right_seq = np.mean(all_right_probs, axis=0)
        avg_rear_seq = np.mean(all_rear_probs, axis=0)

        avg_dash_seq_uniformer_full = np.mean(all_dash_probs_uniformer_full, axis=0)
        avg_right_seq_uniformer_full = np.mean(all_right_probs_uniformer_full, axis=0)
        avg_rear_seq_uniformer_full = np.mean(all_rear_probs_uniformer_full, axis=0)

        avg_dash_seq_x3d = np.mean(all_dash_probs_x3d, axis=0)
        avg_right_seq_x3d = np.mean(all_right_probs_x3d, axis=0)
        avg_rear_seq_x3d = np.mean(all_rear_probs_x3d, axis=0)

        avg_dash_seq = smoothing(np.array(avg_dash_seq), k=1)
        avg_right_seq = smoothing(np.array(avg_right_seq), k=1)
        avg_rear_seq = smoothing(np.array(avg_rear_seq), k=1)

        avg_dash_seq_uniformer_full = smoothing(np.array(avg_dash_seq_uniformer_full), k=1)
        avg_right_seq_uniformer_full = smoothing(np.array(avg_right_seq_uniformer_full), k=1)
        avg_rear_seq_uniformer_full = smoothing(np.array(avg_rear_seq_uniformer_full), k=1)

        avg_dash_seq_x3d = smoothing(np.array(avg_dash_seq_x3d), k=1)
        avg_right_seq_x3d = smoothing(np.array(avg_right_seq_x3d), k=1)
        avg_rear_seq_x3d = smoothing(np.array(avg_rear_seq_x3d), k=1)

        length = min(avg_dash_seq.shape[0], avg_right_seq.shape[0], avg_rear_seq.shape[0])
        avg_dash_seq, avg_right_seq, avg_rear_seq = avg_dash_seq[:length, :], avg_right_seq[:length, :], avg_rear_seq[:length, :]

        length_uniformer_full = min(avg_dash_seq_uniformer_full.shape[0], avg_right_seq_uniformer_full.shape[0], avg_rear_seq_uniformer_full.shape[0])
        avg_dash_seq_uniformer_full, avg_right_seq_uniformer_full, avg_rear_seq_uniformer_full = avg_dash_seq_uniformer_full[:length_uniformer_full, :], avg_right_seq_uniformer_full[:length_uniformer_full, :], avg_rear_seq_uniformer_full[:length_uniformer_full, :]

        length_x3d = min(avg_dash_seq_x3d.shape[0], avg_right_seq_x3d.shape[0], avg_rear_seq_x3d.shape[0])
        avg_dash_seq_x3d, avg_right_seq_x3d, avg_rear_seq_x3d = avg_dash_seq_x3d[:length_x3d, :], avg_right_seq_x3d[:length_x3d, :], avg_rear_seq_x3d[:length_x3d, :]

        prob_ensemble = multi_view_ensemble(avg_dash_seq, avg_right_seq, avg_rear_seq, avg_dash_seq_uniformer_full, avg_right_seq_uniformer_full, avg_rear_seq_uniformer_full, avg_dash_seq_x3d, avg_right_seq_x3d, avg_rear_seq_x3d)
        vid = _FILENAME_TO_ID[right_vid]
        prob_seq = np.array(prob_ensemble)
        prob_seq = np.squeeze(prob_seq)
        clip_classification += activity_localization(prob_seq, vid, 0.3) # default 0.1
        all_model_results[vid] = {"dash": all_dash_probs, "rear": all_rear_probs, "right": all_right_probs}

    clip_classification = pd.DataFrame(clip_classification, columns=["video_id", "label", "start", "end"])
    loc_segments = util_loc.clip_to_segment(clip_classification)
    loc_segments = util_loc.reclassify_segment(loc_segments, all_model_results)
    loc_segments = util_loc.correct_with_prior_constraints(loc_segments)
    with open("pred_ensemble_fold0.txt", "w") as fp:
        for (vid, label, start, end) in loc_segments:
            fp.writelines("{} {} {} {}\n".format(int(vid), label, start, end))
    df = pd.read_csv('pred_ensemble_fold0.txt', delimiter=' ', names=["ID", "Label (Primary)", "Start Time", "End Time"])
    df = df.sort_values(by=['ID', 'Start Time'])
    df.to_csv('pred_ensemble_fold0.txt', index=False, header=False, sep=' ')

if __name__ == "__main__":
    main()