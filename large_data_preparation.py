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
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from fs.osfs import OSFS

import rnn_model_exception

MODEL_DATA_FS = None
RAW_DATA_PATH = 'raw_data/'
RESULT_DATA_PATH = 'result_data/'
RAW_DATA_FILE_EXTENSION = '*.csv'
SEQUENCE_LENGTH = None
FEATURE_SIZE = None

#
def _make_examples(sequence):
    ''' Convert the sequence to examples which length is sequence_length
        If sequence' length is less sequence_length, pad it with zoers
    '''
    _lines = []
    #remove the datetime, and convert string to float
    for line in sequence:
        _line = [float(x) for x in line[1:]]
        _lines.append(_line)
    #Reverse order with datetime
    _lines.reverse()
    #begin to make examples
    _has_short_example = False
    _length_data = len(_lines)
    #At this time, all the example's length is sequence_length
    _num_examples = _length_data // (2*SEQUENCE_LENGTH)
    if _length_data % (2*SEQUENCE_LENGTH) != 0:#There is a short example
        _num_examples += 1
        _has_short_example = True
    #make examples
    _examples = []
    for n in range(_num_examples):
        _example = None
        _target = None
        _start = 2*n*SEQUENCE_LENGTH
        _end = _start + 2*SEQUENCE_LENGTH
        _tmp = None
        if n < _num_examples - 1:
            _tmp = _lines[_start: _end]
            _example = _tmp[0: SEQUENCE_LENGTH]
            _target = _tmp[SEQUENCE_LENGTH:]
        else:
            if _has_short_example:
                _tmp = _lines[_start:]
                _example = _tmp[0: SEQUENCE_LENGTH // 2]
                _target = _tmp[SEQUENCE_LENGTH//2:]
        _examples.append(_example)
        _examples.append(_target)
    return _examples
#
def __write_one_example_to_file__(writer, example):
    for _line in example:
        writer.writerow(_line)
def _save_examples(examples):
    ''' Write the examples to a few file of csv format
        The csv file can have one million lines
        All the lines will dived  sequence of groups of examples
        In a group, the input sequence will be set first, then lines of the target
        For example,

        ......
        22.0, 33.5, 44.3, 22.0, 88.7
        32.2, 22.8, 55.4, 61.3, 99.2
        22.3, 33.5, 67.1, 22.3, 89.9
        33.2, 22.1, 67.4, 33.1, 99.8
        ......

        In above group of example,
        the input sequence is consist of the first two lines,
        and the target sequenc is consist of the esecond tow lines.
        There are other groups before and after the group.
        This function will save the data for train, valid and test
    '''
    #preparation for writer of csv file
    if not isinstance(examples, list):
        raise ValueError("The parameter must be list!")

    _train_csvfile = MODEL_DATA_FS.open(RESULT_DATA_PATH + 'train_data.csv', mode='a')
    _train_writer = csv.writer(_train_csvfile)
    _valid_csvfile = MODEL_DATA_FS.open(RESULT_DATA_PATH +'valid_data.csv', mode='a')
    _valid_writer = csv.writer(_valid_csvfile)
    _test_csvfile = MODEL_DATA_FS.open(RESULT_DATA_PATH + 'test_data.csv', mode='a')
    _test_writter = csv.writer(_test_csvfile)

    for _example in examples:
        _length = len(_example)
        _tmp = []
        if _length != SEQUENCE_LENGTH:
            _padding = np.zeros((SEQUENCE_LENGTH - _length)
                                *FEATURE_SIZE).reshape((SEQUENCE_LENGTH - _length),
                                                       FEATURE_SIZE).tolist()
            _tmp = _example + _padding
        else:
            _temp = _example
        _decision = _make_decision_(90)
        if _decision == 0:
            __write_one_example_to_file__(_train_writer, _tmp)
        if _decision == 1:
            __write_one_example_to_file__(_valid_writer, _tmp)
        if _decision == 2:
            __write_one_example_to_file__(_test_writter, _tmp)
        _train_csvfile.close()
        _valid_csvfile.close()
        _test_csvfile.close()
#
def _combinate_example(total_examples, other_examples):
    if not isinstance(total_examples, list) or not isinstance(other_examples, list):
        raise ValueError("The both two parameter must be a list!")
    for _line in other_examples:
        total_examples.append(_line)
#
def _make_decision_(seed=None):
    _seed = None
    if seed is None:
        _seed = datetime.now().microsecond()
    else:
        _seed = seed
    r_state = np.random.RandomState(_seed)
    _decision = r_state.random_sample()
    if _decision >= 0.0 and _decision < 0.6:#train
        return 0
    if _decision >= 0.6 and _decision < 0.8:#valid
        return 1
    if _decision <= 1.0:#test
        return 2
def _save_prediction_sequence(prediction_sequence):
    ''' This function will save the data for prediction
        the file format is json
    '''
    with MODEL_DATA_FS.open(RESULT_DATA_PATH + 'prediction_sequence.json',
                            mode='w') as jsonfile:
        json.dump(prediction_sequence, jsonfile)
#
def _get_prediction_sequence(filename, sequence):
    return filename, sequence
def _get_statistical_data(examples):
    examples_array = np.array(examples)
    price_array = examples_array[:, 0:4]
    volumn_array = examples_array[:, 4:]
    price_mean = np.mean(price_array)
    price_std = np.std(price_array)
    volumn_mean = np.mean(volumn_array)
    volumn_std = np.std(volumn_array)
    return price_mean, price_std, volumn_mean, volumn_std
def _convert_data_to_example(path, match):
    ''' All the file in raw_pure_path will be converted to the four parts
        The first part is about the examples for train
        The second part is about the example for valid
        The third part is about the example for test
        The fourh part is about the sequence for prediction
        The first three parts will be both input and output sequence
        The last part will be only with input sequence
        The first three parts will be stored in csv format
        The last part will be stored in json for being easy to search
    '''
    files = _get_file_list(path, match)
    if len(list(files)) == 0:
        raise ValueError('There are any files specified by parameter raw_pure_path and match')
    prediction_sequence = {} #store sequence for prediction
    _examples = []#store the first three
    _tmp = []
    for filename in files:
        _lines = []
        with MODEL_DATA_FS.open(filename, mode='r') as _file:
            for _line in _file:
                _lines.append(_line)
        #for prediction
        _key, value = _get_prediction_sequence(filename, _lines)
        prediction_sequence[_key] = value
        _save_prediction_sequence(prediction_sequence)
        #for the first three
        _examples_ = _make_examples(_lines[SEQUENCE_LENGTH:])
        _combinate_example(_tmp, _examples_)

    price_mean, price_std, volumn_mean, volumn_std = _get_statistical_data(_tmp)
    _save_examples(_examples)
#
class stat_feature():
    '''store the stat feature of price and volumn respectively
    '''
    def __init__(self):
        self.num = 0
        self.mean = 0
        self.std = 0
        
#
class example_type():
    train = 0
    valid = 1
    test = 2
#refactor to oriented-object
class InputData():
    ''' This class is for preparation of model data
        At least, this class must convert the csv format to tfrecord format
    '''
    def __init__(self, fsys_data, max_step, feature_size, raw_file_wildcard=None):
        #model parameter
        self.__max_step__ = max_step
        self.__feature_size__ = feature_size
        self.__size_result_file__ = 30*1024*1024
        self.__bits_of_file_number__ = 5
        #file system
        self.__default_raw_data_dir__ = 'raw_data/'
        self.__default_result_data_dir__ = 'result_data/' + 'dataset' + str(max_step) + '/'
        self.__default_log_file__ = 'logerror*.txt'
        self.__fsys_data__ = fsys_data
        #tf.sequenceExample
        self.__example_input_sequence__ = 'input_sequence'
        self.__example_target_sequence__ = 'target_sequence'
        self.__default_token__ = 'token'
        self.__default_sequent_length__ = 'length'
        self.__input_sequence_start_date__ = 'input_start'
        self.__input_sequence_end_date__ = 'input_end'
        self.__target_sequence_start_date__ = 'target_start'
        self.__target_sequence_end_date__ = 'target_end'
        #file match symobles
        #tfrecord file
        self.__default_prediction_tfrecordfile__ = 'prediction*.tfrecord'
        self.__default_valid_tfrecordfile__ = 'valid*.tfrecord'
        self.__default_test_tfrecordfile__ = 'test*.tfrecord'
        self.__default_train_tfrecordfile__ = 'train*.tfrecord'
        #raw data
        if raw_file_wildcard is None:
            self.__raw_file_wildcard__ = '*.csv'
        else:
            self.__raw_file_wildcard__ = raw_file_wildcard
        if self.__fsys_data__.exists(self.__default_result_data_dir__) is not True:
            self.__fsys_data__.makedir(self.__default_result_data_dir__)
        #statistical feature
        self.__stat_price__ = 'price'
        self.__stat_volumn__ = 'volumn'
        self._stat_features = {self.__stat_price__: stat_feature(),
                               self.__stat_volumn__: stat_feature()}
    #
    def __setup_result_dir__(self):
        def __createneededfiles__(match):
            fsys = self.__fsys_data__
            pure_path = self.__default_result_data_dir__
            filename = ''
            file_name = match.split('*')
            if match == self.__default_log_file__:
                filename = file_name[0] + file_name[1]
                if fsys.exists(pure_path + filename) is not True:
                    with fsys.open(pure_path + filename, mode="w") as logerror:
                        writer = csv.writer(logerror)
                        writer.writerow(['file_name', 'error_type'])
            else:
                filename = file_name[0] + self.__bits_of_file_number__*'0' + file_name[1]
                if fsys.exists(pure_path + filename) is not True:
                    fsys.create(pure_path + filename)

        fsys = self.__fsys_data__
        pure_path = self.__default_result_data_dir__
        if fsys.exists(pure_path) is not True:
            fsys.makedir(pure_path)
        __createneededfiles__(self.__default_log_file__)
        __createneededfiles__(self.__default_prediction_tfrecordfile__)
        __createneededfiles__(self.__default_test_tfrecordfile__)
        __createneededfiles__(self.__default_train_tfrecordfile__)
        __createneededfiles__(self.__default_valid_tfrecordfile__)

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
        files = self._get_files(self.__default_raw_data_dir__, self.__raw_file_wildcard__)
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
            _example_type = _make_decision_type()
            if _example_type == example_type.train:
                self._calculate_statistical_feature(example)
            ''' At this place, the encoded example must be stored in tfrecord files
            '''
            ''' At this place, it must be determined
                that an example is used for training, testing or verification.
                Such as for training, to calculate its statistical characteristic values
            '''
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

        input_sequence = example[self.__example_input_sequence__]
        target_sequence = example[self.__example_target_sequence__]
        #remove the data of exchange date
        input_lines = [line[1: self.__feature_size__ + 1 ] for line in input_sequence]
        target_lines = [line[1:self.__feature_size__ + 1] for line in target_sequence]
        
        lines = np.array(input_lines + target_lines)
        
        stat_price = stat_feature()
        stat_volumn = stat_feature()
        
        stat_price.num = len(input_lines + target_lines) * (self.__feature_size__ - 1)
        price_lines = lines[0:, 0:self.__feature_size__ - 1]
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
        context_sequent_length = self.__default_sequent_length__
        context_token = self.__default_token__
        context_start = self.__input_sequence_start_date__
        context_end = self.__input_sequence_end_date__
        input_sequence = self.__example_input_sequence__

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
        context_features = {self.__default_token__:
                                tf.FixedLenFeature([], dtype=tf.string),
                            self.__default_sequent_length__:
                                tf.FixedLenFeature([], dtype=tf.int64),
                            self.__input_sequence_end_date__:
                                tf.FixedLenFeature([], dtype=tf.string),
                            self.__input_sequence_start_date__:
                                tf.FixedLenFeature([], dtype=tf.string)
                           }
        sequence_features = {self.__example_input_sequence__:
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
        input_sequence = example_lines[self.__example_input_sequence__]
        target_sequence = example_lines[self.__example_target_sequence__]
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
            self.__default_sequent_length__].int64_list.value.append(length)
        ex.context.feature[
            self.__input_sequence_start_date__].bytes_list.value.append(input_start)
        ex.context.feature[
            self.__input_sequence_end_date__].bytes_list.value.append(input_end)
        ex.context.feature[
            self.__target_sequence_start_date__].bytes_list.value.append(target_start)
        ex.context.feature[
            self.__target_sequence_end_date__].bytes_list.value.append(target_end)
        ex.context.feature[
            self.__default_token__].bytes_list.value.append(token_bytes)
        fl_inputs = ex.feature_lists.feature_list[self.__example_input_sequence__]
        fl_targets = ex.feature_lists.feature_list[self.__example_target_sequence__]
        for _input, _target in zip(inputs, targets):
            fl_inputs.feature.add().float_list.value.append(_input)
            fl_targets.feature.add().float_list.value.append(_target)
        return ex
    #
    def _decode_train_example(self, ex_serial):
        context_features = {
            self.__default_sequent_length__: tf.FixedLenFeature([], dtype=tf.int64),
            self.__input_sequence_start_date__: tf.FixedLenFeature([], dtype=tf.string),
            self.__input_sequence_end_date__: tf.FixedLenFeature([], dtype=tf.string),
            self.__target_sequence_start_date__: tf.FixedLenFeature([], dtype=tf.string),
            self.__target_sequence_end_date__: tf.FixedLenFeature([], dtype=tf.string),
            self.__default_token__: tf.FixedLenFeature([], dtype=tf.string)
        }
        sequence_features = {
            self.__example_input_sequence__: tf.FixedLenSequenceFeature([], dtype=tf.float32),
            self.__example_target_sequence__: tf.FixedLenSequenceFeature([], dtype=tf.float32)
        }
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=ex_serial,
            context_features=context_features,
            sequence_features=sequence_features
        )
        return context_parsed, sequence_parsed
        
    #
    def __create_filename__(self, files):
        ''' The form of name of result data are "prediction000.tfrecord"
            The number in name of file is the "000"
            When the size is large enough, the result data will be stored in another file
            And the name of new file will be the "prediction001.tfrecord"
            The method will search the in the list of files, and find the greatest number
            Then will create an new file which number will be plus 1 to the greatest number
        '''
        _file = files[len(files) - 1]
        _filename = _file.split('.')
        file_base = _filename[0][0:-self.__bits_of_file_number__]
        file_suffix = _filename[1]
        file_number = int(_filename[0][-self.__bits_of_file_number__:])
        file_number += 1
        string_number = None
        if file_number >= 100: string_number = str(file_number)
        if file_number < 100 and  file_number >= 10: string_number = '0' + str(file_number)
        if file_number < 10: string_number = '00' +str(file_number)
        filename = file_base + string_number + '.' + file_suffix
        return filename
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
        with self.__fsys_data__.open(self.__default_raw_data_dir__ + filename,
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
        path = self.__default_result_data_dir__ + self.__default_log_file__
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
