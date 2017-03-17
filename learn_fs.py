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
    pure_path = pathlib.Path('data/result_data/')
    filename = 'some.csv'
    with myfs.open(str(pure_path) + filename, mode='w') as myfile:
        writer = csv.writer(myfile)
        for line in examples:
            writer.writerow(line)
    with myfs.open(str(pure_path) + filename, mode='r') as myfile:
        reader = csv.reader(myfile)
        lines = []
        for line in reader:
            _line = [int(x) for x in line]
            lines.append(_line)
        print(lines)
        