import csv
import json
import tempfile

from fs.memoryfs import MemoryFS
import numpy as np

fsys = MemoryFS()
fsys.makedir('raw_data')
fsys.makedir('result_data')
fsys.makedir('result_data/dataset_200_step/')
pure_path = 'result_data/dataset_200_step/'
fsys.create(pure_path + 'a.tfrecord')
fsys.create(pure_path + 'b.tfrecord')
fsys.create(pure_path + 'log.log')
fsys.tree()
fsys.removetree(pure_path)
fsys.tree()