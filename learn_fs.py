import csv
import json
import tempfile

import fs
import numpy as np
from fs.memoryfs import MemoryFS
from fs.opener import open_fs

fsys = fs.open_fs('data')
path1 = 'result_data'
path2 = 'result_data/dataset200_step'
filename = 'result_data/dataset200_step/test.tfrecord'
file_exist = fsys.exists(filename)
if fsys.exists(filename) is True:
    fsys.remove(filename)
fsys.removedir(path2)
fsys.removedir(path1)