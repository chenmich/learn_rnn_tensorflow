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
from datetime import datetime
import json
import csv
import argparse
import numpy as np
from fs.osfs import OSFS

MODEL_DATA_FS = None
RAW_DATA_PATH = 'raw_data/'
RESULT_DATA_PATH = 'result_data/'
RAW_DATA_FILE_EXTENSION = '*.csv'
SEQUENCE_LENGTH = 0


def _get_file_list(fsys, pure_path, match):
    ''' return all the file by matching parameter match in pure_path' folder
        Arg:
            pure_path: pure path without file name
            match:file name with wildcard, for example, *.csv, some*.csv
    '''
    _filelist = list(fsys.filterdir(pure_path, files=[match]))
    filelist = [x.name for x in _filelist]
    return filelist
#
def _get_prediction_sequence(filename, sequence_length, sequence):
    ''' From sequence extracte the prediction sequence which length is sequence_length
        From filename extracte the key which is file's name without suffix
    '''
    if len(sequence) < sequence_length:
        raise ValueError("The seqence is short for demand!")
    _key = filename.split('.')[0]
    _prediction_sequence = sequence[0:sequence_length]
    _prediction_sequence.reverse()
    return _key, _prediction_sequence
#
def _make_examples(sequence_length, sequence):
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
    _num_examples = _length_data // (2*sequence_length)
    if _length_data % (2*sequence_length) != 0:#There is a short example
        _num_examples += 1
        _has_short_example = True
    #make examples
    _examples = []
    for n in range(_num_examples):
        _example = None
        _target = None
        _start = 2*n*sequence_length
        _end = _start + 2*sequence_length
        _tmp = None
        if n < _num_examples - 1:
            _tmp = _lines[_start: _end]
            _example = _tmp[0: sequence_length]
            _target = _tmp[sequence_length:]
        else:
            if _has_short_example:
                _tmp = _lines[_start:]
                _example = _tmp[0: sequence_length // 2]
                _target = _tmp[sequence_length//2:]
        _examples.append(_example)
        _examples.append(_target)
    return _examples
#
def _save_examples(fsys, examples):
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
    '''_csvfile = fsys.open('data/result_data/' + 'train_data.csv', mode='a')
    _train_writer = csv.writer(_csvfile)
    _csvfile = fsys.open()'''
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
def _save_prediction_sequence(fsys, prediction_sequence):
    ''' This function will save the data for prediction
        the file format is json
    '''
    with fsys.open(RESULT_DATA_PATH + 'prediction_sequence.json',
                   mode='w') as jsonfile:
        json.dump(prediction_sequence, jsonfile)
#

def _convert_data_to_example(fsys, sequence_length, path, match):
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
    files = _get_file_list(MODEL_DATA_FS, path, match)
    if len(list(files)) == 0:
        raise ValueError('There are any files specified by parameter raw_pure_path and match')
    prediction_sequence = {} #store sequence for prediction
    _examples = []#store the first three
    for filename in files:
        _lines = []
        with fsys.open(filename, mode='r') as _file:
            for _line in _file:
                _lines.append(_line)
        #for prediction
        _key, value = _get_prediction_sequence(filename, sequence_length, _lines)
        prediction_sequence[_key] = value
        _save_prediction_sequence(fsys, prediction_sequence)
        #for the first three
        _examples_ = _make_examples(sequence_length, _lines[sequence_length:])
        _combinate_example(_examples, _examples_)
        _save_examples(fsys, _examples)


# main control
def main(args):
    ''' main control flow
    '''
    SEQUENCE_LENGTH = ARGS.Sequence_length
    MODEL_DATA_FS = OSFS(args.data_path)
    MODEL_DATA_FS.tree()
    #_convert_data_to_example(args.Sequence_length, args.data_path + RAW_DATA_PATH, RAW_DATA_FILE_EXTENSION)
    MODEL_DATA_FS.close()
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        help="data path",
                        type=str,
                        default='data/')
    parser.add_argument("--convert", "-Con", action='store_true',
                        help="convert all the raw file to fileexamples")
    parser.add_argument("--Sequence_length", "-Length",
                        default=200,
                        help='the sequence length in a example')
    ARGS = parser.parse_args()
    main(ARGS)
