import os
import glob
import pandas as pd
import numpy as np

path_A1 = os.listdir('A1')
list_avg1, list_avg2, list_avg3, list_avg4, list_avg5, list_avg6, list_avg7, list_avg8, list_avg9, list_avg10, list_avg11, list_avg12, list_avg13, list_avg14, list_avg15 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(len(path_A1)):
	list_csv = glob.glob('A1/' + path_A1[i] + '/*.csv')
	for j in range(len(list_csv)):
		df = pd.read_csv(list_csv[j], delimiter=',')
		df['Start Time'] = pd.to_timedelta(df['Start Time']).dt.total_seconds().astype(int)
		df['End Time'] = pd.to_timedelta(df['End Time']).dt.total_seconds().astype(int)
		df['Time Difference'] = df['End Time'] - df['Start Time']
		mean_time_difference_class_1 = df[df['Label (Primary)'] == 'Class 1']['Time Difference'].mean()
		mean_time_difference_class_2 = df[df['Label (Primary)'] == 'Class 2']['Time Difference'].mean()
		mean_time_difference_class_3 = df[df['Label (Primary)'] == 'Class 3']['Time Difference'].mean()
		mean_time_difference_class_4 = df[df['Label (Primary)'] == 'Class 4']['Time Difference'].mean()
		mean_time_difference_class_5 = df[df['Label (Primary)'] == 'Class 5']['Time Difference'].mean()
		mean_time_difference_class_6 = df[df['Label (Primary)'] == 'Class 6']['Time Difference'].mean()
		mean_time_difference_class_7 = df[df['Label (Primary)'] == 'Class 7']['Time Difference'].mean()
		mean_time_difference_class_8 = df[df['Label (Primary)'] == 'Class 8']['Time Difference'].mean()
		mean_time_difference_class_9 = df[df['Label (Primary)'] == 'Class 9']['Time Difference'].mean()
		mean_time_difference_class_10 = df[df['Label (Primary)'] == 'Class 10']['Time Difference'].mean()
		mean_time_difference_class_11 = df[df['Label (Primary)'] == 'Class 11']['Time Difference'].mean()
		mean_time_difference_class_12 = df[df['Label (Primary)'] == 'Class 12']['Time Difference'].mean()
		mean_time_difference_class_13 = df[df['Label (Primary)'] == 'Class 13']['Time Difference'].mean()
		mean_time_difference_class_14 = df[df['Label (Primary)'] == 'Class 14']['Time Difference'].mean()
		mean_time_difference_class_15 = df[df['Label (Primary)'] == 'Class 15']['Time Difference'].mean()
		list_avg1.append(mean_time_difference_class_1)
		list_avg2.append(mean_time_difference_class_2)
		list_avg3.append(mean_time_difference_class_3)
		list_avg4.append(mean_time_difference_class_4)
		list_avg5.append(mean_time_difference_class_5)
		list_avg6.append(mean_time_difference_class_6)
		list_avg7.append(mean_time_difference_class_7)
		list_avg8.append(mean_time_difference_class_8)
		list_avg9.append(mean_time_difference_class_9)
		list_avg10.append(mean_time_difference_class_10)
		list_avg11.append(mean_time_difference_class_11)
		list_avg12.append(mean_time_difference_class_12)
		list_avg13.append(mean_time_difference_class_13)
		list_avg14.append(mean_time_difference_class_14)
		list_avg15.append(mean_time_difference_class_15)
		
		#for i in range(len(df)):
			#if not pd.isna(df.at[i, 'Filename']):
				#last_non_nan_value = df.at[i, 'Filename']
			#df.at[i, 'Filename'] = last_non_nan_value
		#save_path = str(list_csv[j].split('/')[0]) + '/' + str(list_csv[j].split('/')[1]) + '/' + str(list_csv[j].split('/')[-1])
		#df.to_csv(save_path, index=False)
		
		#print(df.columns[df.isnull().any()])
		
		#equal_start_end_rows = df[df['Start Time'] == df['End Time']]
		#if not equal_start_end_rows.empty:
			#print("File:", list_csv[j])
			#print(equal_start_end_rows)
			#print("\n")

list_mean = [np.nanmean(np.array(list_avg1)), np.nanmean(np.array(list_avg2)), np.nanmean(np.array(list_avg3)),
			 np.nanmean(np.array(list_avg4)), np.nanmean(np.array(list_avg5)), np.nanmean(np.array(list_avg6)),
			 np.nanmean(np.array(list_avg7)), np.nanmean(np.array(list_avg8)), np.nanmean(np.array(list_avg9)),
			 np.nanmean(np.array(list_avg10)), np.nanmean(np.array(list_avg11)), np.nanmean(np.array(list_avg12)),
			 np.nanmean(np.array(list_avg13)), np.nanmean(np.array(list_avg14)), np.nanmean(np.array(list_avg15))]
list_max = [np.nanmax(np.array(list_avg1)), np.nanmax(np.array(list_avg2)), np.nanmax(np.array(list_avg3)),
			 np.nanmax(np.array(list_avg4)), np.nanmax(np.array(list_avg5)), np.nanmax(np.array(list_avg6)),
			 np.nanmax(np.array(list_avg7)), np.nanmax(np.array(list_avg8)), np.nanmax(np.array(list_avg9)),
			 np.nanmax(np.array(list_avg10)), np.nanmax(np.array(list_avg11)), np.nanmax(np.array(list_avg12)),
			 np.nanmax(np.array(list_avg13)), np.nanmax(np.array(list_avg14)), np.nanmax(np.array(list_avg15))]
list_min = [np.nanmin(np.array(list_avg1)), np.nanmin(np.array(list_avg2)), np.nanmin(np.array(list_avg3)),
			 np.nanmin(np.array(list_avg4)), np.nanmin(np.array(list_avg5)), np.nanmin(np.array(list_avg6)),
			 np.nanmin(np.array(list_avg7)), np.nanmin(np.array(list_avg8)), np.nanmin(np.array(list_avg9)),
			 np.nanmin(np.array(list_avg10)), np.nanmin(np.array(list_avg11)), np.nanmin(np.array(list_avg12)),
			 np.nanmin(np.array(list_avg13)), np.nanmin(np.array(list_avg14)), np.nanmin(np.array(list_avg15))]
list_std = [np.nanstd(np.array(list_avg1)), np.nanstd(np.array(list_avg2)), np.nanstd(np.array(list_avg3)),
			 np.nanstd(np.array(list_avg4)), np.nanstd(np.array(list_avg5)), np.nanstd(np.array(list_avg6)),
			 np.nanstd(np.array(list_avg7)), np.nanstd(np.array(list_avg8)), np.nanstd(np.array(list_avg9)),
			 np.nanstd(np.array(list_avg10)), np.nanstd(np.array(list_avg11)), np.nanstd(np.array(list_avg12)),
			 np.nanstd(np.array(list_avg13)), np.nanstd(np.array(list_avg14)), np.nanstd(np.array(list_avg15))]
list_median = [np.nanmedian(np.array(list_avg1)), np.nanmedian(np.array(list_avg2)), np.nanmedian(np.array(list_avg3)),
			 np.nanmedian(np.array(list_avg4)), np.nanmedian(np.array(list_avg5)), np.nanmedian(np.array(list_avg6)),
			 np.nanmedian(np.array(list_avg7)), np.nanmedian(np.array(list_avg8)), np.nanmedian(np.array(list_avg9)),
			 np.nanmedian(np.array(list_avg10)), np.nanmedian(np.array(list_avg11)), np.nanmedian(np.array(list_avg12)),
			 np.nanmedian(np.array(list_avg13)), np.nanmedian(np.array(list_avg14)), np.nanmedian(np.array(list_avg15))]
dict = {'Class': np.arange(1, 16), 'Mean': list_mean, 'Median': list_median, 'Max': list_max, 'Min': list_min, 'Std': list_std}	
final_df = pd.DataFrame(dict)
print(final_df)