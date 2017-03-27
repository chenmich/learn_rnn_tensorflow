import csv
with open('data/raw_data/some00000.csv', mode='r') as csvfile:
    reader = csv.reader(csvfile)
    print(next(reader))
    print()
    print('=======================================================')
    print(next(reader))

def my_generator():
    for x in  range(10000):
        yield x

y = my_generator()
print()
print(next(y))
print()
print('========================================================')
print(next(y))