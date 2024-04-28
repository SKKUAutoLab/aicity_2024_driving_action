import os
import json

name_file = os.listdir('aaa')
list_train = name_file[:54]
list_val = name_file[54:]
dict_train = {"train": list_train, "val": list_val}
with open('data_split_0.json', 'w') as f:
    json.dump(dict_train, f)
