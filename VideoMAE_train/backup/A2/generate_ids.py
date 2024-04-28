import pandas as pd
import re
import numpy as np
import os

df = pd.DataFrame()
# df['video_id'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# df['video_files1'] = ['Dashboard_user_id_26223_NoAudio_3.MP4', 'Dashboard_user_id_26223_NoAudio_5.MP4', 'Dashboard_user_id_42897_NoAudio_5.MP4',
#                       'Dashboard_user_id_42897_NoAudio_7.MP4', 'Dashboard_user_id_49989_NoAudio_5.MP4', 'Dashboard_user_id_49989_NoAudio_7.MP4',
#                       'Dashboard_user_id_51953_NoAudio_3.MP4', 'Dashboard_user_id_51953_NoAudio_7.MP4', 'Dashboard_user_id_96715_NoAudio_5.MP4', 'Dashboard_user_id_96715_NoAudio_7.MP4']
# df['video_files2'] = ['Rear_view_user_id_26223_NoAudio_3.MP4', 'Rear_view_user_id_26223_NoAudio_5.MP4', 'Rear_view_user_id_42897_NoAudio_5.MP4',
#                       'Rear_view_user_id_42897_NoAudio_7.MP4', 'Rear_view_user_id_49989_NoAudio_5.MP4', 'Rear_view_user_id_49989_NoAudio_7.MP4',
#                       'Rear_view_user_id_51953_NoAudio_3.MP4', 'Rear_view_user_id_51953_NoAudio_7.MP4', 'Rear_view_user_id_96715_NoAudio_5.MP4', 'Rear_view_user_id_96715_NoAudio_7.MP4']
# df['video_files3'] = ['Right_side_window_user_id_26223_NoAudio_3.MP4', 'Right_side_window_user_id_26223_NoAudio_5.MP4', 'Right_side_window_user_id_42897_NoAudio_5.MP4',
#                       'Right_side_window_user_id_42897_NoAudio_7.MP4', 'Right_side_window_user_id_49989_NoAudio_5.MP4', 'Right_side_window_user_id_49989_NoAudio_7.MP4',
#                       'Right_side_window_user_id_51953_NoAudio_3.MP4', 'Right_side_window_user_id_51953_NoAudio_7.MP4', 
#                       'Right_side_window_user_id_96715_NoAudio_5.MP4', 'Right_side_window_user_id_96715_NoAudio_7.MP4']
df['video_id'] = np.arange(1, 31)
list_dashboard = []
list_rearview = []
list_rightside = []
path_A2 = os.listdir('A2/')
for i in range(len(path_A2)):
    if os.path.isdir('A2/' + path_A2[i]):
        list_file = os.listdir('A2/' + path_A2[i])
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

df.to_csv("video_ids.csv", index=False, header=True, sep=',')
videos = pd.read_csv('video_ids.csv')
for idx, row_data in videos.iterrows():
    user_id = re.search("user_id_\d{5}", row_data[1])[0]
    print(user_id, 'aaa')
    print(row_data[1], 'bbb')
    print(row_data[2], 'ccc')
    print(row_data[3], 'ddd')