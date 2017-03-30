somestring = 'python12'
x= somestring[-2:]
x_int = int(x)
x_int += 1
new_string = somestring[0:-2] + str(x_int)
print(new_string)

print('=================================================')
files = ['some000.csv', 'some001.csv', 'some002.csv', 'some003.csv', 'some004.csv',
        'some000.csv']
_file = files[len(files) - 1]
_filename = _file.split('.')
file_base = _filename[0][0:-3]
file_suffix = _filename[1]
file_number = int(_filename[0][-3:])
file_number += 1
string_number = None
if file_number >= 100: string_number = str(file_number)
if file_number < 100 and  file_number >= 10: string_number = '0' + str(file_number)
if file_number < 10: string_number = '00' +str(file_number)
filename = file_base + string_number + '.' + file_suffix
print(filename)