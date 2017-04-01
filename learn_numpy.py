import numpy as np
import csv
import datetime
from datetime import date
import itertools
import data_reader as dr
from datetime import time



_lines = [[1, 2, 3, 4, 110],[22, 55, 44, 88, 99]]

lines = [[str(datetime.date.today())] + _line for _line in _lines]
line = [line[1:] for line in lines]
print(line)