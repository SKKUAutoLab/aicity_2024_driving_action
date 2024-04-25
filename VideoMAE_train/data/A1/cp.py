import os
import shutil

list_csv = os.listdir('A1')
for i in range(len(list_csv)):
    name_file = list_csv[i].split('.')[0]
    shutil.move('A1/' + name_file + '.csv', 'aaa/' + name_file + '/' + name_file + '.csv')