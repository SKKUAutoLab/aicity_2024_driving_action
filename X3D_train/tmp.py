import os
import glob
import pandas as pd

# path = os.listdir('aaa')
# for file in range(len(path)):
#     path_mp4 = glob.glob('aaa/' + path[file] + '/' + '*.MP4')
#     for filename in path_mp4:
#         if 'Rearview' in filename:
#             os.rename(filename, filename.replace('Rearview', 'Rear_view'))

# file_path = os.listdir('2022/A1')
# for k in range(len(file_path)):
#     path = glob.glob('2022/A1/' + str(file_path[k]) + '/*.csv')
#     for i in range(len(path)):
#         df = pd.read_csv(path[i])
#         if 'Filename' in df.columns:
#             #df['Filename'] = df['Filename'].str.replace('Rearview_', 'Rear_view_')
#             df['Filename'] = df['Filename'].str.replace('Dashboard_User', 'Dashboard_user')
#             df['Filename'] = df['Filename'].str.replace('Rear_view_User', 'Rear_view_user')
#             df['Filename'] = df['Filename'].str.replace('Right_side_window_User', 'Right_side_window_user')
#             df.to_csv(path[i], index=False)
#             print(f'Renamed "Rearview" to "Rear_view" in {path[i]}')

from natsort import natsorted

list_labels = []
list_videos = []
for i in range(16):
    path_videos = natsorted(glob.glob('data/' + str(i) + '/' + '*.MP4'))
    for j in range(len(path_videos)):
        list_videos.append(path_videos[j])
        list_labels.append(int(path_videos[j].split('/')[1]))
df = pd.DataFrame({'Filename': list_videos, 'Class': list_labels})
df.to_csv('total_data.csv', header=False, index=False, sep=' ')
