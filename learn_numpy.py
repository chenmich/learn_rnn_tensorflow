import numpy as np
import csv
import datetime
from datetime import date
import itertools
import data_reader as dr
from datetime import time



_lines = [[1, 2, 3, 4, 110],[22, 55, 44, 88, 99]]

lines_array = np.array(_lines)
price_lines = lines_array[0:, 0:3]
volumn_lines = lines_array[0:, 4:]


another_lines_array = np.arange(10).reshape(2,5)

three_lines = np.sta
print(three_lines)