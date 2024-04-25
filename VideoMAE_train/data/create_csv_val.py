import pandas as pd
import os

global_id_counter = 1
filename_ids = {}
path_user = sorted(os.listdir('Val_Fold_4'))
dfs = []
for i in range(len(path_user)):
    if path_user[i].split('.')[-1] == 'csv':
        continue
    else:
        df = pd.read_csv('Val_Fold_4/' + path_user[i] + '/' + path_user[i] + '.csv', sep=',')
        df1 = df[df['Filename'].str.startswith('Dashboard')]
        # df2 = df1[['Filename', 'Label (Primary)']].copy()
        df2 = df1[['Filename', 'Label (Primary)', 'Start Time', 'End Time']].copy()
        start_time_lst = list(df2['Start Time'].values)
        end_time_lst = list(df['End Time'].values)
        for i in range(len(df2)):
        	ftr = [3600,60,1]
        	df2['Start Time'][i] = sum([a*b for a,b in zip(ftr, map(int, start_time_lst[i].split(':')))])
        	df2['End Time'][i] = sum([a*b for a,b in zip(ftr, map(int,end_time_lst[i].split(':')))])
        
        df2 = df2[df2['Label (Primary)'] != 'Class 0']
        filename_ids = {}
        df2['ID'] = df2.groupby('Filename').ngroup() + global_id_counter
        global_id_counter += df2['ID'].nunique()
        dfs.append(df2)

result_df = pd.concat(dfs)
# final_df = result_df[['ID', 'Label (Primary)']].copy()
final_df = result_df[['ID', 'Label (Primary)', 'Start Time', 'End Time']].copy()

final_df['Label (Primary)'] = final_df['Label (Primary)'].str.split().str[-1]
final_df.to_csv('gt_val_fold4.txt', index=False, header=False, sep=' ')
