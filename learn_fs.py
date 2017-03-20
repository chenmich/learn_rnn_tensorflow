import csv
import json
from fs.memoryfs import MemoryFS
from fs.opener import open_fs
import fs
import numpy as np

home_local = MemoryFS()
home_local.makedir('data')
home_local.close()
examples = np.arange(10000).reshape(2000, 5).tolist()
with MemoryFS() as myfs:
    myfs.makedir('data')
    myfs.makedir('data/raw_data')
    myfs.makedir('data/result_data')
    pure_path = 'data/result_data/'
    filename = 'some.csv'
    path = pure_path + filename
    with myfs.open(path, mode='w') as myfile:
        writer = csv.writer(myfile)
        for line in examples:
            writer.writerow(line)
    path = pure_path + 'another.csv'
    with myfs.open(path, mode='w') as anotherfile:
        writer = csv.writer(anotherfile)
        for line in examples:
            writer.writerow(line)
    with myfs.open(pure_path + filename, mode='r') as myfile:
        reader = csv.reader(myfile)
        lines = []
        for line in reader:
            _line = [int(x) for x in line]
            lines.append(_line)
    myfs.tree()
    myfs.listdir(pure_path)
    files = list(myfs.scandir('data/result_data/'))
    print(files[0].name)
    files = list(myfs.filterdir('data/result_data/', files=['some*.csv']))
    print(files[0].name)
#
print('____________________')
_fsys = open_fs('data/')
filelist = list(_fsys.filterdir('/raw_data/', files=["*.csv"]))
for _file in filelist:
    print(_file.name)

some = {"ms":[1,2,3],
        "ns":[11,12,18]}
with MemoryFS() as anotherfs:
    anotherfs.makedir('data')
    with anotherfs.open('data/some.json', mode='w') as jsonfile:
        json.dump(some, jsonfile)
    with anotherfs.open('data/some.json', mode='r') as jsonfile:
        somejson = json.load(jsonfile)
        print(somejson)

#
filename = 'some00000.csv'
str_filename = filename.split(".data")
print(str_filename)
with fs.osfs.OSFS('.') as _fsys:
    _anoter_fsys = _fsys.opendir('data/')
    print()
    _fsys.tree()
    print()
    _anoter_fsys.tree()
    _anoter_fsys.close()



model_data_fs = fs.memoryfs.MemoryFS()
model_data_fs.makedir('data')
model_data_fs.makedir('data/raw_data')
model_data_fs.makedir('data/result_data')
_raw_data = np.arange(20000).reshape(4000, 5).tolist()
filenames = ['some00000.csv', 'some00001.csv', 'some00002.csv',
                'some00003.csv', 'some00004.csv']

for _file in filenames:
    with model_data_fs.open('data/raw_data/' + _file, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        for line in _raw_data:
            writer.writerow(line)
print("___________________________________________")
print('____________________________________________')
model_data_fs.tree()
_other_fsys = model_data_fs.opendir('data/result_data')
print("another")
_other_fsys.tree()