import numpy as np
import csv
import datetime
from datetime import date

lines = [[1, 2, 3, 4, 110],
         [2, 3, 4, 5, 222],
         [3, 4, 5, 6, 333]]
lines_array = np.array(lines, dtype='float16')
lines_array_slice = lines_array[:,0:4]
mean = np.mean(lines_array_slice)
std = np.std(lines_array_slice)
lines_array_slice -= mean
lines_array_slice_norm = np.divide(lines_array_slice, std)


another_slice = lines_array[0:, 4:]
#some = np.reshape(another_slice, 3)
another_mean = np.mean(another_slice)
another_std = np.std(another_slice)
another_slice -= another_mean
another_slice_norm = np.divide(another_slice, std)

lines_array_norm = np.hstack((lines_array_slice_norm, another_slice_norm))



x = np.arange(16.0).reshape(4, 4)
y1, y2 = np.hsplit(x, indices_or_sections=[3,])
z1 = x[0:1]
z2 = x[1:]
print(z1)
print()
print(z2)
std = np.std(y2)
#print(lines.reverse())

b = np.arange(24).reshape((2, 3, 4))
b[::-1,]
c = b[0]
c[::-1,]



