import glob
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

list_train = ["user_id_53307", "user_id_20090", "user_id_35418", "user_id_87903", "user_id_86356", "user_id_61497", "user_id_72446", "user_id_46856", "user_id_59014", "user_id_85870", "user_id_61302", "user_id_71436", "user_id_36305"]
list_dirs = []
list_clips = []
list_train.sort()
for i in range(16): # 16 activities
    path_split = glob.glob('A1_clip/' + str(i) + '/*.MP4')
    list_clips.append(path_split)
list_clips = flatten(list_clips)
for i in range(len(list_train)):
    for j in range(len(list_clips)):
        if list_train[i] in list_clips[j]:
            list_dirs.append(list_clips[j])
list_pd_dirs_right_side = []
list_pd_labels_right_side = []
list_pd_dirs_rear_view = []
list_pd_labels_rear_view = []
list_pd_dirs_dashboard = []
list_pd_labels_dashboard = []
for i in range(len(list_dirs)):
    if 'Right_side' in list_dirs[i]:
        list_pd_dirs_right_side.append(list_dirs[i])
        list_pd_labels_right_side.append(list_dirs[i].split('/')[1])
    if 'Rear_view' in list_dirs[i]:
        list_pd_dirs_rear_view.append(list_dirs[i])
        list_pd_labels_rear_view.append(list_dirs[i].split('/')[1])
    if 'Dashboard' in list_dirs[i]:
        list_pd_dirs_dashboard.append(list_dirs[i])
        list_pd_labels_dashboard.append(list_dirs[i].split('/')[1])
df1 = pd.DataFrame()
df1['dir'] = list_pd_dirs_right_side
df1['dir'] = 'data/' + df1['dir']
df1['label'] = list_pd_labels_right_side 
df1.to_csv("test_right_0.csv", index=False, header=False, sep=' ')

df2 = pd.DataFrame()
df2['dir'] = list_pd_dirs_rear_view
df2['dir'] = 'data/' + df2['dir']
df2['label'] = list_pd_labels_rear_view
df2.to_csv("test_rear_0.csv", index=False, header=False, sep=' ')

df3 = pd.DataFrame()
df3['dir'] = list_pd_dirs_dashboard
df3['dir'] = 'data/' + df3['dir']
df3['label'] = list_pd_labels_dashboard 
df3.to_csv("test_dash_0.csv", index=False, header=False, sep=' ')
