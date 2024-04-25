import pandas as pd
import os

list_dashboard = []
list_rearview = []
list_rightside = []
path_A2 = os.listdir('A2/')
for i in range(len(path_A2)):
    if os.path.isdir('A2/' + path_A2[i]):
        list_file = os.listdir('A2/' + path_A2[i])
        for j in range(len(list_file)):
            if 'Dashboard' in list_file[j]:
                list_dashboard.append(list_file[j].split('.')[0])
            elif 'Rear_view' in list_file[j]:
                list_rearview.append(list_file[j].split('.')[0])
            else:
                list_rightside.append(list_file[j].split('.')[0])
list_dashboard = sorted(list_dashboard)
list_rearview = sorted(list_rearview)
list_rightside = sorted(list_rightside)
dict1 = {item: index + 1 for index, item in enumerate(list_dashboard)}
dict2 = {item: index + 1 for index, item in enumerate(list_rearview)}
dict3 = {item: index + 1 for index, item in enumerate(list_rightside)}
merged_dict = {**dict2, **dict1, **dict3}
print(merged_dict)
