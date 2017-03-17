import csv
import csv
import pathlib
from fs.memoryfs import MemoryFS
from fs.opener import open_fs
import numpy as np

home_local = MemoryFS()
home_local.makedir('data')
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
    