import pandas as pd
import re
import numpy as np
import os
from argparse import ArgumentParser

def create_ids(args):
    num_videos_test = len(os.listdir(args.test_path))
    df = pd.DataFrame()
    df['video_id'] = np.arange(1, (num_videos_test * 2) + 1)
    list_dashboard = []
    list_rearview = []
    list_rightside = []
    path_A2 = os.listdir(args.test_path)
    for i in range(len(path_A2)):
        if os.path.isdir(args.test_path + path_A2[i]):
            list_file = os.listdir(args.test_path + path_A2[i])
            for j in range(len(list_file)):
                if 'Dashboard' in list_file[j]:
                    list_dashboard.append(list_file[j])
                elif 'Rear_view' in list_file[j]:
                    list_rearview.append(list_file[j])
                else:
                    list_rightside.append(list_file[j])
    list_dashboard = sorted(list_dashboard)
    list_rearview = sorted(list_rearview)
    list_rightside = sorted(list_rightside)
    df['video_files1'] = list_dashboard
    df['video_files2'] = list_rearview
    df['video_files3'] = list_rightside
    df.to_csv(args.test_path + "video_ids.csv", index=False, header=True, sep=',')
    videos = pd.read_csv(args.test_path + 'video_ids.csv')
    for idx, row_data in videos.iterrows():
        user_id = re.search("user_id_\d{5}", row_data[1])[0]
        print(user_id, row_data[1], row_data[2], row_data[3])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_path', type=str, default='B/')
    args = parser.parse_args()
    create_ids(args)