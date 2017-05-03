import csv
import fs
class someFiles():
    def __init__(fsys):
        self.fsys = fsys
    def get_lines(self):
        raw_file_names = self.get_files(self.__raw_data_dir__, '*.csv')
        lines = []
        for raw_file_name in raw_file_names:
            with self.__fsys_data__.open(self.__raw_data_dir__ + raw_file, mode='r') as raw_file:
                reader = csv.reader(raw_file)
                for line in reader:
                    lines.append(line)
            if self._is_comptible(lines) is False: break
            yield raw_file_name, lines
            lines = []

def get_fsys():
    ''' simulate the file system with pyfilesystem
        pyfilesystem implement the general interface of file system.
        Replace the built-in file system with pyfilesystem to improve
        dependence injection
    '''
    model_data_fs = memoryfs.MemoryFS()
    model_data_fs.makedir('data')
    model_data_fs.makedir('data/raw_data')
    model_data_fs.makedir('data/result_data')
    _raw_data = np.random.normal(loc=0.0, scale=1.0, size=20000).reshape(4000, 5).tolist()
    filenames = ['some00000.csv', 'some00001.csv', 'some00002.csv',
                 'some00003.csv', 'some00004.csv']

    for _file in filenames:
        with model_data_fs.open('data/raw_data/' + _file, mode='w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['data', 'open', 'close', 'max', 'min', 'volumn'])
            for line in _raw_data:
                writer.writerow([str(datetime.date.today())] + line)
    return model_data_fs.opendir('data')

fsys = get_fsys()
some = someFiles(fsys)
