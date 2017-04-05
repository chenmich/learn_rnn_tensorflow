import csv
import json
from fs.memoryfs import MemoryFS
from fs.opener import open_fs
import fs
import numpy as np
import tempfile

x = tempfile._RandomNameSequence()
for sequence in x:
    print(sequence)
    y = input()
    if y == 'ok':
        break
