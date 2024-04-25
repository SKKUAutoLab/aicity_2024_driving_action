import os
import glob
import pandas as pd

# path = os.listdir('aaa')
# for file in range(len(path)):
#     path_mp4 = glob.glob('aaa/' + path[file] + '/' + '*.MP4')
#     for filename in path_mp4:
#         if 'Rearview' in filename:
#             os.rename(filename, filename.replace('Rearview', 'Rear_view'))

path = glob.glob('A1/*.csv')
for i in range(len(path)):
    df = pd.read_csv(path[i])
    if 'Filename' in df.columns:
        df['Filename'] = df['Filename'].str.replace('Rearview_', 'Rear_view_')
        df.to_csv(path[i], index=False)
        print(f'Renamed "Rearview" to "Rear_view" in {path[i]}')