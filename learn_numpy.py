import numpy as np
import csv
import data_reader as dr
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
std = np.std(y2)
#print(lines.reverse())

b = np.arange(24).reshape((2, 3, 4))
b[::-1,]
c = b[0]
c[::-1,]



with open("some.csv", mode='r') as some:
    reader = csv.reader(some, dialect='excel')
    line_list = []
    first_line = next(reader)
    for line in reader:
        _line = [float(_x) for _x in line[1:]]
        line_list.append(_line)

with open("some.csv", mode='r') as dictSoem:
    reader = csv.DictReader(dictSoem, fieldnames=["date", "open", "close", "max", "min", "value"])
    first_line = next(reader)
    second_line = next(reader)
    
exchange_date = second_line['date']
print(exchange_date)
exchange_date = datetime.datetime.strptime("30-Sep-08", '%d-%b-%y')
print(exchange_date)

