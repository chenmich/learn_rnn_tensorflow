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
''' Test large-data-preparation module
'''
import json
import csv
import numpy as np
from fs import memoryfs
import large_data_preparation as ldp
import tensorflow as tf

#for all test class
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
    _raw_data = np.arange(20000).reshape(4000, 5).tolist()
    filenames = ['some00000.csv', 'some00001.csv', 'some00002.csv',
                 'some00003.csv', 'some00004.csv']

    for _file in filenames:
        with model_data_fs.open('data/raw_data/' + _file, mode='w') as csvfile:
            writer = csv.writer(csvfile)
            for line in _raw_data:
                writer.writerow(line)
    return model_data_fs.opendir('data')

#
MAX_STEP = 200
FEATURE_SIZE = 5
MODEL_DATA_FS = get_fsys()
FILE_WILDCARD = '*.csv'
NUM_RAW_FILES = 5
#

#test InputData class
class test_get_raw_data_files(tf.test.TestCase):
    ''' test the method of InputData' method _get_raw_data_files
    '''
    def test_returned_value(self):
        ''' valid returned value
        '''
        inputdata = ldp.InputData(MODEL_DATA_FS, MAX_STEP, FEATURE_SIZE)
        files = inputdata._get_raw_data_files(FILE_WILDCARD)
        self.assertEqual(len(files), NUM_RAW_FILES)
        self.assertEqual(files[0], 'some00000.csv')
        self.assertEqual(files[1], 'some00001.csv')
        self.assertEqual(files[2], 'some00002.csv')
        self.assertEqual(files[3], 'some00003.csv')
        self.assertEqual(files[4], 'some00004.csv')
#
class test_content_not_enough(tf.test.TestCase):
    '''test the method of _content_not_enough
    '''
    pass
#
class test_make_examples(tf.test.TestCase):
    '''test the method _make_examples
    '''
    def test_for_content_not_enough(self):
        with MODEL_DATA_FS.open('some00006.csv', mode='w') as raw_file:
            writer = csv.writer(raw_file)
            lines = np.arange(MAX_STEP - 1, FEATURE_SIZE)


if __name__ == "__main__":
    tf.test.main()
    MODEL_DATA_FS.close()
    print('ok!')
