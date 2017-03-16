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
''' Generate a serie of parabolic mapping
'''
import numpy as np
import csv

def non_linear_parabolic_curve_map_data_reader(num_batch=1000, batch_size=5,
                                               sequence_length=200, feature_size=5,
                                               mu=1.401155189, init_value=0.618):
    ''' generate a na.array object wiht shape = [batch_size, sequence_length, feature_size]
        Arg:
            num_batch:number of batch about data_reader
            batch_size:quantity of batches
            sequence_length: length of series
            feature_size:feature number of one data_reader
            mu:parameter of parabolic mapping
            init_value:initial value for mapping
    '''
    for _ in range(num_batch):
        batch_data = __non_linear_parabolic_curve_map_generate_data__(2*batch_size*sequence_length*feature_size,
                                                                      mu=mu, init_value=init_value)#one dimension data
        #cut up to two series
        batch_data_x = batch_data[0: batch_size*sequence_length*feature_size]
        batch_data_y = batch_data[batch_size*sequence_length*feature_size:]
        # reshape to (batch_size, sequence_length, feature_size)
        # from batch_size*sequence_length*feature_size
        x = np.reshape(batch_data_x, (batch_size, sequence_length, feature_size))
        y = np.reshape(batch_data_y, (batch_size, sequence_length, feature_size))
        yield x, y



#
def __non_linear_parabolic_curve_map_generate_data__(num, mu=1.401155189, init_value=0.618):
    x = init_value
    #set the first 200 data aside
    for _ in range(200):
        x = 1 - mu*x*x
    #generate all the data
    X = []
    for _ in range(num):
        x = 1 - mu*x*x
        X.append(x)
    return np.array(X)
#
def sinFun_data_reader(num_batch=1000, batch_size=5,
                       sequence_length=200, feature_size=5):
    for _ in range(num_batch):
        x = np.random.randn(batch_size, sequence_length, feature_size) 
        y = x*np.sin(x)
        yield x, y
#

class SomeData():
    def __init__(self, files, sequence_length=200, batch_size=5):
        if not isinstance(files, list):
            raise TypeError("The parameter files must be a list")
        if len(files) == 0:
            raise ValueError("The files is Null!")
        self._files = files
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._price_mean = 0
        self._price_std = 0
        self._volumn_mean = 0
        self._volumn_std = 0
        self._train_data = []
        self._valid_data = []
        for _filename in self._files:
            with open(_filename, mode='r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                _train = []
                _valid = []
                n = 0
                for _line in reader:
                    _tmp = _line[0]
                    _line_number = [float(x) for x in _line[1:]]
                    _line = [_tmp] + _line_number
                    if n < sequence_length:
                        _valid.append(_line)
                    else:
                        _train.append(_line)
                    n += 1
                _train.reverse()
                _valid.reverse()
                self._train_data.append(_train)
                self._valid_data.append(_valid)
    #
    def train_data(self):
        ''' generate example
        '''
        _has_short_example = False
        for _data in self._train_data:
            _length_data = len(_data)
            _num_examples = _length_data // (2* self._sequence_length)
            if _length_data % (2*self._sequence_length) == 0:
                _num_examples -= 1
                _has_short_example = True
            examples = []
            targets = []
            for n in range(_num_examples):
                _example = None
                _target = None
                if n < _num_examples:
                    _example = _data[n*self._sequence_length: (n + 1)*self._sequence_length]
                    _target = _data[(n + 1)*self._sequence_length: (n + 2)*self._sequence_length]
                else:
                    if _has_short_example:
                        _example = _data[n*self._sequence_length/2: (n + 1)*self._sequence_length/2]
                        _target = _data[(n + 1)*self._sequence_length / 2:]
                examples.append(_example)
                targets.append(_target)


