y = ['1.0', '2.0']
print(y)
def foo(x):
   length = len(x)
   for no in range(length):
       x[no] = float(x[0])
foo(y)
print(y)