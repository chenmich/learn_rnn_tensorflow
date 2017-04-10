import csv
import json
from fs.memoryfs import MemoryFS
from fs.opener import open_fs
import fs
import numpy as np
import tempfile
'''
x = tempfile._RandomNameSequence()
for sequence in x:
    print(sequence)
    y = input()
    if y == 'ok':
        break
'''
fsys = fs.memoryfs.MemoryFS()
fsys.makedir('data/')
fsys.makedir('data/result_data/')
#fsys.create('data/result_data/prediction_aldshjls.tfrecord')
fsys.tree()
with fsys.open('data/result_data/prediction_aldshjls.tfrecord', mode='a') as fp:
    writer = csv.writer(fp)
    writer.writerow([1, 2, 3, 4, 5])
fsys.tree()
info = fsys.getdetails('data/result_data/prediction_aldshjls.tfrecord')
print(info)
