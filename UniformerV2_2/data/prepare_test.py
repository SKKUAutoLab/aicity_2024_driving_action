import glob
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]

list_train = ["user_id_86952", "user_id_96371", "user_id_22640", "user_id_69039", "user_id_33682", "user_id_40908", "user_id_98389", "user_id_39367", "user_id_30932", "user_id_42711", "user_id_50347", "user_id_63764", "user_id_83756", "user_id_60167", "user_id_99882", "user_id_46844", "user_id_63513", "user_id_38159", "user_id_84935", "user_id_70176", "user_id_79755", "user_id_13522", "user_id_46141", "user_id_71720", "user_id_20507", "user_id_83323", "user_id_28557", "user_id_99660", "user_id_16080", "user_id_93439", "user_id_25432", "user_id_38479", "user_id_60768", "user_id_50921", "user_id_93542", "user_id_79023", "user_id_31903", "user_id_59581", "user_id_14786"]
list_dirs = []
list_clips = []
list_train.sort()
list_activities = [0, 11, 12, 13]
for i in range(len(list_activities)): # 4 activities
    path_split = glob.glob('A1_clip_custom/' + str(list_activities[i]) + '/*.MP4')
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
df1.to_csv("val_right_4.csv", index=False, header=False, sep=' ')

df2 = pd.DataFrame()
df2['dir'] = list_pd_dirs_rear_view
df2['dir'] = 'data/' + df2['dir']
df2['label'] = list_pd_labels_rear_view
df2.to_csv("val_rear_4.csv", index=False, header=False, sep=' ')

df3 = pd.DataFrame()
df3['dir'] = list_pd_dirs_dashboard
df3['dir'] = 'data/' + df3['dir']
df3['label'] = list_pd_labels_dashboard 
df3.to_csv("val_dash_4.csv", index=False, header=False, sep=' ')
