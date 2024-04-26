import glob
import os
from natsort import natsorted
import pandas as pd

path_csv = natsorted(glob.glob('data/*.csv'))
for i in range(len(path_csv)):
    df = pd.read_csv(path_csv[i], names=['video_ids', 'class'], sep=' ')
    df['video_ids'] = df['video_ids'].apply(lambda x: x.replace("11/Right", "1/Right").replace("12/Right", "2/Right").replace("13/Right", "3/Right"))
    # df['class'] = df['class'].apply(lambda x: 1 if x == 11 else (2 if x == 12 else (3 if x == 13 else x)))
    df.to_csv(path_csv[i], header=False, index=False, sep=' ')
