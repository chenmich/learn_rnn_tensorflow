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
std = np.std(y2)
#print(lines.reverse())

b = np.arange(24).reshape((2, 3, 4))
b[::-1,]
c = b[0]
c[::-1,]

f = [
    [
        [111, 112, 113],
        [121, 122, 123],
        [131, 132, 133]
        ],
    [
        [211, 212, 213],
        [221, 222, 223, 224]
        ]
    ]
f_array = np.array(f)
g_array = np.reshape(f_array, [2, -1])
print(g_array.shape)
print(g_array)
print(f_array.shape)



