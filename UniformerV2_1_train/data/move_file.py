import os
import glob
import shutil
from natsort import natsorted

path_A1 = os.listdir('A1_clip')
for i in range(len(path_A1)):
    if path_A1[i] == '11' or path_A1[i] == '12' or path_A1[i] == '13':
        path_sub_A1 = natsorted(glob.glob('A1_clip/' + path_A1[i] + '/*.MP4'))
        for j in range(len(path_sub_A1)):
            path_cls = path_sub_A1[j].split('/')[1]
            path_name = path_sub_A1[j].split('/')[-1]
            shutil.copy(path_sub_A1[j], 'A1_clip_custom/' + path_cls + '/' + path_name)
    else:
        path_sub_A1 = natsorted(glob.glob('A1_clip/' + path_A1[i] + '/*.MP4'))
        for j in range(len(path_sub_A1)):
            path_name = path_sub_A1[j].split('/')[-1]
            shutil.copy(path_sub_A1[j], 'A1_clip_custom/0/' + path_name)