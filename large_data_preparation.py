# Copyright 2017 The Chenmich Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' This module make examples from sequence data
'''
import argparse
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf

import fs
from fs.osfs import OSFS

import rnn_model_exception

#
class stat_feature():
    '''store the stat feature of price and volumn respectively
    '''
    def __init__(self):
        self.num = 0
        self.mean = 0
        self.std = 0

#
class ExampleType():
    ''' the type code of examples based on the usage 
    '''
    train = 0
    valid = 1
    test = 2
    prediction = 3
#
class AllForFile():
    ''' This class is a container for string about file
    '''
    prediction_record = 'prediction*.tfrecord'
    valid_record = 'valid*.tfrecord'
    test_record = 'test*.tfrecord'
    train_record = 'train*.tfrecord'
    raw_file_wildcard = '*.csv'
    log_file = 'logerror*.txt'
    # size of the all tfrecord file, if larger than this, it will be divided
    result_file_size = 30 * 1024 * 1024
#
class ExampleString():
    ''' This class for example's string
    '''
    input_sequence = 'input_sequence'
    target_sequence = 'target_sequence'
    token = 'token'
    sequent_length = 'length'
    input_start_date = 'input_start'
    input_end_date = 'input_end'
    target__start_date = 'target_start'
    target_end_date = 'target_end'
#
class RawDataLine():
    def __init__(self, token, lines):
        self.token = token
        self.lines = lines
example_map = {ExampleType.prediction: AllForFile.prediction_record,
               ExampleType.test: AllForFile.test_record,
               ExampleType.train: AllForFile.train_record,
               ExampleType.valid: AllForFile.valid_record}
#
class FilesOp():
    '''This class's responsiblity is for save and get examples
    '''
    def __init__(self, fsys_data):
        fsys = fsys_data
        self.__fsys_data__ = fsys_data
        self.__result_data_dir__ = 'result_data/'
        self.__raw_data_dir__ = 'raw_data/'
        if self.__has_raw_data() is False:
            raise  rnn_model_exception.NoRawDataFileFound('There is any file of raw data!')
        if(fsys.exists(self.__result_data_dir__)) is False:
            fsys.makedir(self.__result_data_dir__)
        self.__serialized_examples__ = []
    def __has_raw_data(self):
        ''' check whether there are any files of raw data
        '''
        fsys = self.__fsys_data__
        has_raw_data = True
        if(fsys.exists(self.__raw_data_dir__)) is False:
            has_raw_data = False
        filelist = self.get_files(self.__raw_data_dir__, '*.csv')
        if filelist.__len__ == 0:
            has_raw_data = False
        return has_raw_data

    def get_files(self, pure_path, match):
        ''' This method get all the files in specified raw data dir
        '''
        _filelist = list(self.__fsys_data__.filterdir(pure_path,
                                                      files=[match]))
        filelist = [x.name for x in _filelist]
        return filelist
    #
    def empty_default_result_dir(self):
        ''' remove all the content of a directory(include that delete all sub directories)
        '''
        fsys = self.filesystem()
        directory = self.default_result_dir()
        if fsys.exists(directory) is False:
            raise fs.errors.DestinationExists('The directory ' + directory +' is not exist!')
        info = fsys.getinfo(directory, 'details',)
        if info.is_dir is False:
            raise fs.errors.DirectoryExpected('The ' + directory + 'is not a directory!')
        fsys.removetree(directory)
        fsys.makedir(directory)
    #
    @property
    def default_result_dir(self):
        ''' the property of default result directory
        '''
        return self.__default_result_dir__
    #
    @default_result_dir.setter
    def default_result_dir(self, default_result_dir):
        fsys = self.__fsys_data__
        if fsys.exists(self.__result_data_dir__ + default_result_dir) is False:
            fsys.makedir(self.__result_data_dir__ + default_result_dir)

    @property
    def filesystem(self):
        ''' filesystem'''
        return self.__fsys_data__
    #
    def save_examples(self, ex_serial):
        ''' save an serialized example to a tf.record file
        '''
        self.__serialized_examples__.append(ex_serial)
        if self.__examples_len__() >= AllForFile.result_file_size:
            self.__save_examples__()
    #
    def __save_examples__(self):
        pass
    #
    def __examples_len__(self):
        __len = 0
        for example in self.__serialized_examples__:
            __len += example.__len__()
        return __len
    def get_raw_lines(self):
        ''' Get the raw lines of example from files of raw data
            return:
                token:file name without extensive name of raw data
                lines:the raw lines        
        '''
        fsys = self.__fsys_data__
        raw_file_names = self.get_files(self.__raw_data_dir__, '*.csv')        
        for raw_file_name in raw_file_names:

            with fsys.open(self.__raw_data_dir__ + raw_file_name, mode='r') as raw_file:
                reader = csv.reader(raw_file)
                lines = []
                for line in reader:
                    lines.append(line)
                raw_lines = RawDataLine(raw_file_name, lines[1:])
                yield raw_lines
    #
    
            
#refactor to oriented-object
class InputData():
    ''' This class is for preparation of model data
        At least, this class must convert the csv format to tfrecord format
    '''
    def __init__(self, fsys_data, max_step, feature_size):
        #model parameter
        self.__max_step__ = max_step
        self.__feature_size__ = feature_size
        if isinstance(files, Files) is False:
            raise TypeError('The argument files must be the instance of class Files')
        else:
            self.__files__ = files
            self.__files__.default_result_dir('result_data/' + 'dataset_' + (
                str(max_step) + '_step' + '/'))
        
        #file system
        self.__raw_data_dir__ = 'raw_data/'
        self.__result_data_dir__ = 'result_data/' + 'dataset_' + (
            str(max_step) + '_step' + '/')
        self.__fsys_data__ = fsys_data
        #statistical feature
        self.__stat_price__ = 'price'
        self.__stat_volumn__ = 'volumn'
        self._stat_features = {self.__stat_price__: stat_feature(),
                               self.__stat_volumn__: stat_feature()}
        self.__setup_result_dir__()
    #
    def __setup_result_dir__(self):
        fsys = self.__fsys_data__
        pure_path = self.__result_data_dir__
        if fsys.exists('/result_data/') is not True:
            fsys.makedir('/result_data/')
        if fsys.exists(pure_path) is not True:
            fsys.makedir(pure_path)
    #
    def make_examples(self):
        ''' This method will divided raw to four parts
            The first part of the parts is for train of model
            The second part of the parts is for valid of model
            The third part of the parts is for test of model
            The fouth of the parts will be used to predict the future value
            And The first three parts are unified as training data
        '''
        fsys = self.__fsys_data__
        files = self._get_files(self.__raw_data_dir__, AllForFile.raw_file_wildcard)
        if len(files) == 0:
            raise rnn_model_exception.NoRawDataFileFound('There are any files of raw data found')
        #randomly shuffle the order of files
        files_array = np.array(files)
        np.random.shuffle(files_array)
        files = files_array.tolist()
        prediction_examples = {}
        _lines = []
        for raw_file in files:
            # if comptible, data converted to float will be stored in _lines
            # after _raw_data_check is called
            is_comptible, lines = self._raw_data_check(raw_file)
            if is_comptible:
                token = raw_file.split('.')[0]
                #there may be several file of ticker for raw data.
                # I will name it ticker_000.csv, ticker_001.csv and so on
                token = token.split('_')[0]
                # In the raw data file, the data line
                # for the later exchange date is at the forefront
                # I will reverse the order
                _lines.reverse()
                length = len(_lines)
                self._make_examples_for_prediction(lines[length - self.__max_step__],
                                                   token)
                self._make_examples_for_trains(lines[length - self.__max_step__:],
                                               token)
            else:#logerror for file of raw data
                pass
    #
    def _make_examples_for_prediction(self, lines, token):
        ''' The raw data will be divided to four parts. one of the four parts is for prediction
            for real price
            This method is make all the examples for prediction
            args:
                lines: The all lines of one of all the raw files.
                       The first max_step lines is for prediction
                file_name: file name with extension of a raw file
                           the file name is key dictionary for examples for prediction
        '''
        ex = self._encode_prediction_example(lines, token)
        self._save_examples(ex.SerializeToString(), ExampleType.prediction)

    #
    def _make_examples_for_trains(self, lines, token):
        ''' The examples for train are consist of the three parts.
            The first part is for train. The second part is for valid for fine adjustment
            of the hyperparameter. And the third part is for test of model
            This method will divide the lines into the three parts
            This method will save all the examples, so there is no return values
            args:
                lines: all the raw data in list
        '''
        def _make_decision_type(seed=None):
            _seed = None
            if seed is None:
                _seed = datetime.now().microsecond()
            else:
                _seed = seed
            r_state = np.random.RandomState(_seed)
            _decision = r_state.random_sample()
            if _decision >= 0.0 and _decision < 0.6:#train
                return example_type.train
            if _decision >= 0.6 and _decision < 0.8:#valid
                return example_type.valid
            if _decision <= 1.0:#test
                return example_type.test

        #divide the lines for example
        examples = self._divide_line(lines)
        for example in examples:
            ex = self._encode_train_example(example, token)
            _example_type = _make_decision_type(seed=0)
            if _example_type == ExampleType.train:
                self._calculate_statistical_feature(example)
            self._save_examples(ex.SerializeToString(), _example_type)
    #
    def _save_examples(self, ex_serial, ex_type):
        filename = self._get_fileobject(ex_type)
        with self.__fsys_data__.open(filename, mode='a') as fp:
            writer = tf.python_io.TFRecordWriter(fp.name)
            writer.write(ex_serial)
            writer.close()
    #
    def _get_fileobject(self, ex_type):
        pass
    #
    def _calculate_statistical_feature(self, example):
        def _combinate_stat(stat1, stat2):
            ''' This function will calculate the statistic value
                when the number of data, mean and std of two batch of data are available.
                The equations are:
                e = a1*mean1 + a2*mean2
                sigma^2 = a1*(sigma1^2 + e1^2) + a2*(sigma2^2 + e2^2) - e*e
                a1 = n1 / (n1 + n2)
                a2 = n2 / (n1 + n2)
            '''
            n1 = stat1.num
            n2 = stat2.num
            a1 = n1 / (n1 + n2)
            a2 = n2 / (n1 + n2)
            e = a1 * stat1.mean + a2 * stat2.mean
            sigma_square = a1*(np.square(stat1.std) + stat1.mean * stat1.mean)
            sigma_square += a2*(np.square(stat2.std) + stat2.mean * stat2.mean) - e * e

            stat = stat_feature()
            stat.num = n1 + n2
            stat.mean = e
            stat.std = np.sqrt(sigma_square)
            return stat

        input_sequence = example[ExampleString.input_sequence]
        target_sequence = example[ExampleString.target_sequence]
        #remove the data of exchange date
        input_lines = [line[1: self.__feature_size__ + 1 ] for line in input_sequence]
        target_lines = [line[1:self.__feature_size__ + 1] for line in target_sequence]
        #join the two sequence for calculate stat
        lines = np.array(input_lines + target_lines)

        stat_price = stat_feature()
        stat_volumn = stat_feature()


        price_lines = lines[0:, 0:self.__feature_size__ - 1]
        stat_price.num = len(input_lines + target_lines) * (self.__feature_size__ - 1)
        stat_price.mean = np.mean(price_lines)
        stat_price.std = np.std(price_lines)

        volumn_lines = lines[0:, self.__feature_size__ - 1 :]
        stat_volumn.num = len(input_lines + target_lines)
        stat_volumn.mean = np.mean(volumn_lines)
        stat_volumn.std = np.std(volumn_lines)

        #With the existing stat combination
        combinate_stat_price = _combinate_stat(stat_price, self._stat_features[self.__stat_price__])
        combinate_stat_volumn = _combinate_stat(stat_volumn, self._stat_features[self.__stat_volumn__])
        self._stat_features[self.__stat_price__] = combinate_stat_price
        self._stat_features[self.__stat_volumn__] = combinate_stat_volumn
    #
    def _divide_line(self, raw_data_lines):
        ''' this method will divide the lines of raw data to line of examples
            args:
                lines: line of raw data
                return: list of line of example
        '''
        #begin to make examples
        max_step = self.__max_step__
        has_short_example = False
        length = len(raw_data_lines)
        #assume that at this time, all the example's length is self.__max_step__
        num_examples = length // (2*max_step)
        if length % (2*max_step) != 0:#There is a short example
            num_examples += 1
            has_short_example = True
        #make examples
        examples = []
        for n in range(num_examples):
            input_sequence = []
            target_sequence = []
            _start = 2*n*max_step
            _end = _start + 2*max_step
            _tmp = []
            if n < num_examples - 1:
                _tmp = raw_data_lines[_start: _end]
                input_sequence = _tmp[0: max_step]
                target_sequence = _tmp[max_step:]
            else:
                if has_short_example:
                    _tmp = raw_data_lines[_start:]
                    input_sequence = _tmp[0: max_step // 2]
                    target_sequence = _tmp[max_step//2:]
            examples.append({'input_sequence': input_sequence,
                             'target_sequence': target_sequence})
        return examples


    #
    def _encode_prediction_example(self, lines, token):
        #proccess the lines
        length = self.__max_step__
        _token = bytes(token, 'utf-8')
        #note:the data of date of exchange are tried to keeped as much as posssible
        # In the raw data file, the data line for the later exchange date is at the forefront
        start_line = lines[0]
        start = bytes(start_line[0], 'utf-8')
        end_line = lines[length - 1]
        end = bytes(end_line[0], 'utf-8')
        #divide the price and date of exchange
        _lines = [line[1:] for line in lines]
        #get the lines flatten to store
        sequence = np.array(_lines)
        sequence = sequence.flatten().tolist()

        #prepared for ex
        context_sequent_length = ExampleString.sequent_length
        context_token = ExampleString.token
        context_start = ExampleString.input_start_date
        context_end = ExampleString.input_end_date
        input_sequence = ExampleString.input_sequence

        #build ex
        ex = tf.train.SequenceExample()
        ex.context.feature[context_sequent_length].int64_list.value.append(length)
        ex.context.feature[context_token].bytes_list.value.append(_token)
        ex.context.feature[context_start].bytes_list.value.append(start)
        ex.context.feature[context_end].bytes_list.value.append(end)
        input_feature = ex.feature_lists.feature_list[input_sequence]
        for x in sequence:
            input_feature.feature.add().float_list.value.append(x)
        return ex
    #
    def _decode_prediction_example(self, ex_serial):
        context_features = {ExampleString.token:
                                tf.FixedLenFeature([], dtype=tf.string),
                            ExampleString.sequent_length:
                                tf.FixedLenFeature([], dtype=tf.int64),
                            ExampleString.input_end_date:
                                tf.FixedLenFeature([], dtype=tf.string),
                            ExampleString.input_start_date:
                                tf.FixedLenFeature([], dtype=tf.string)
                           }
        sequence_features = {ExampleString.input_sequence:
                             tf.FixedLenSequenceFeature([], dtype=tf.float32)
                            }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex_serial,
            context_features=context_features,
            sequence_features=sequence_features
            )
        return context_parsed, sequence_parsed
    #
    def _encode_train_example(self, example_lines, token):
        #prepare content
        #get the two sequence
        input_sequence = example_lines[ExampleString.input_sequence]
        target_sequence = example_lines[ExampleString.target_sequence]
        #remove data of exchange date and get them flatten
        _inputs = [line[1:] for line in input_sequence]
        _targets = [line[1:] for line in target_sequence]
        inputs = np.array(_inputs).flatten()
        targets = np.array(_targets).flatten()
        #basic data
        length = len(input_sequence)
        input_start = bytes(input_sequence[0][0], 'utf-8')
        input_end = bytes(input_sequence[length-1][0], 'utf-8')
        target_start = bytes(target_sequence[0][0], 'utf-8')
        target_end = bytes(target_sequence[length - 1][0], 'utf-8')
        token_bytes = bytes(token, 'utf-8')
        #begin to encode
        ex = tf.train.SequenceExample()
        ex.context.feature[
            ExampleString.sequent_length].int64_list.value.append(length)
        ex.context.feature[
            ExampleString.input_start_date].bytes_list.value.append(input_start)
        ex.context.feature[
            ExampleString.input_end_date].bytes_list.value.append(input_end)
        ex.context.feature[
            ExampleString.target__start_date].bytes_list.value.append(target_start)
        ex.context.feature[
            ExampleString.target_end_date].bytes_list.value.append(target_end)
        ex.context.feature[
            ExampleString.token].bytes_list.value.append(token_bytes)
        fl_inputs = ex.feature_lists.feature_list[ExampleString.input_sequence]
        fl_targets = ex.feature_lists.feature_list[ExampleString.target_sequence]
        for _input, _target in zip(inputs, targets):
            fl_inputs.feature.add().float_list.value.append(_input)
            fl_targets.feature.add().float_list.value.append(_target)
        return ex
    #
    def _decode_train_example(self, ex_serial):
        context_features = {
            ExampleString.sequent_length: tf.FixedLenFeature([], dtype=tf.int64),
            ExampleString.input_start_date: tf.FixedLenFeature([], dtype=tf.string),
            ExampleString.input_end_date: tf.FixedLenFeature([], dtype=tf.string),
            ExampleString.target__start_date: tf.FixedLenFeature([], dtype=tf.string),
            ExampleString.target_end_date: tf.FixedLenFeature([], dtype=tf.string),
            ExampleString.token: tf.FixedLenFeature([], dtype=tf.string)
        }
        sequence_features = {
            ExampleString.input_sequence: tf.FixedLenSequenceFeature([], dtype=tf.float32),
            ExampleString.target_sequence: tf.FixedLenSequenceFeature([], dtype=tf.float32)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex_serial,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return context_parsed, sequence_parsed

    #
    def _get_files(self, pure_path, match):
        ''' This method get all the files in specified raw data dir
        '''
        _filelist = list(self.__fsys_data__.filterdir(pure_path,
                                                      files=[match]))
        filelist = [x.name for x in _filelist]
        return filelist
    #
    def _raw_data_check(self, filename):
        # This method's resposibilities are to make sure that number of lines is enough
        # and that the needed string can be converted to number
        # If it is not comptible, this method log it
        ''' process the problem that the lines of one file are less max_step!
            process the porble that some data of price and volumn can be converted float
            args:
                lines: lines of data. It is a list
        '''
        _is_comptible = True
        lines = []
        with self.__fsys_data__.open(self.__raw_data_dir__ + filename,
                                     mode='r') as raw_file:

            reader = csv.reader(raw_file)
            next(reader)
            for line in reader:
                try:
                    tmp = line[0]
                    line_data = line[1:]
                    tmp_data = [float(x) for x in line_data]
                    #Try to keep date data as much as possible
                    lines.append([tmp] + tmp_data)
                except ValueError:
                    _is_comptible = False
                    #log
                    self.__log_data_not_comptible__(
                        filename, rnn_model_exception.DataNotComptible.has_non_float)
                    return _is_comptible
        length = len(lines)
        if length < self.__max_step__:
            _is_comptible = False
            #log
            self.__log_data_not_comptible__(filename,
                                            rnn_model_exception.DataNotComptible.is_not_enough)
        return _is_comptible, lines
    #
    def __log_data_not_comptible__(self, filename, error_type):
        def __createlogerrorfile__(path):
            with self.__fsys_data__.open(path, mode='w') as logerror:
                writer = csv.writer(logerror)
                writer.writerow(['filename', 'errorlog'])
        path = self.__result_data_dir__ + AllForFile.log_file
        if self.__fsys_data__.exists(path) is not True:
            __createlogerrorfile__(path)
        with self.__fsys_data__.open(path, mode='a') as logfile:
            writer = csv.writer(logfile)
            writer.writerow([filename, error_type])
    #
# main control
def main(args):
    ''' main control flow
    '''
    fsys_for_data = OSFS(args.data_path)
    inputdata = InputData(fsys_for_data, args.max_step, args.feature_size)
    inputdata.make_examples()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        help="data path",
                        type=str,
                        default='data/')
    parser.add_argument("--convert", "-Con", action='store_true',
                        help="convert all the raw file to fileexamples")
    parser.add_argument("--max_step", "-MS",
                        default=200,
                        help='the Max number for time step in a example')
    parser.add_argument("--feature_size", "-F_size",
                        help='The number of feature in one example',
                        default=5)
    parser.add_argument("--file_wildcard", "-FE", help="the wildcard name of raw data file",
                        default='*.csv')
    ARGS = parser.parse_args()
    main(ARGS)
