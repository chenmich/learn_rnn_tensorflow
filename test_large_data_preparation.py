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
import csv
import datetime
import numpy as np
import tensorflow as tf
from fs import memoryfs

import large_data_preparation as ldp
import rnn_model_exception


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
            writer.writerow(['data', 'open', 'close', 'max', 'min', 'volumn'])
            for line in _raw_data:
                writer.writerow([datetime.datetime.now()] + line)
    return model_data_fs.opendir('data')

#
MAX_STEP = 200
FEATURE_SIZE = 5
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
        fsys = get_fsys()
        inputdata = ldp.InputData(fsys, MAX_STEP, FEATURE_SIZE)
        files = inputdata._get_raw_data_files()
        self.assertEqual(len(files), NUM_RAW_FILES)
        self.assertEqual(files[0], 'some00000.csv')
        self.assertEqual(files[1], 'some00001.csv')
        self.assertEqual(files[2], 'some00002.csv')
        self.assertEqual(files[3], 'some00003.csv')
        self.assertEqual(files[4], 'some00004.csv')
        fsys.close()

#
class test_make_example(tf.test.TestCase):
    ''' This class test method _make_examples
    '''
    def setUp(self):
        self.fsys = get_fsys()
    def tearDown(self):
        self.fsys = get_fsys()
    #
    def test_make_examples_for_prediction_called(self):
        # preparation for test
        class InputDataForTest_Pred(ldp.InputData):
            self._make_examples_for_prediction_called = False
            def _raw_data_check(self, filename, lines):
                # force to return a True fot to test if _make_examples will call the method
                return True
            def _make_examples_for_prediction(self, lines, tokens):
                self._make_examples_for_prediction_called = True
            # separat the effect of this method
            def _make_training_examples(self, lines, tokens):
                pass
        inputdata = InputDataForTest_Pred(self.fsys, MAX_STEP, FEATURE_SIZE)
        #exercise
        inputdata.make_examples()
        # only if the method _make_examples_for_prediction is called,
        # the assertion will be correction
        self.assertTrue(self.test_make_examples_for_prediction_called)
    #
    def test_make_examples_for_train_called(self):
        # preparation for test
        class InputDataForTest_Train(ldp.InputData):
            self._make_training_examples_called = False
            def _raw_data_check(self, filename, lines):
                # force to return a True fot to test if _make_examples will call the method
                return True
            def _make_examples_for_prediction(self, lines, tokens):
                pass
            def _make_training_examples(self, lines, tokens):
                self._make_training_examples_called = True
        inputdata = InputDataForTest_Train(self.fsys, MAX_STEP, FEATURE_SIZE)
        inputdata.make_examples()
        self.assertTrue(inputdata._make_training_examples_called)
#
class test_raw_data_check(tf.test.TestCase):
    def setUp(self):
        self.fsys = get_fsys()
    def tearDown(self):
        self.fsys.close()
    def test_raw_data_check_True(self):
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        lines = []
        self.assertTrue(inputdata._raw_data_check('some00000.csv', lines))
    #
    def test_raw_data_check_Length_Not_Enough(self):
        lines = np.arange((MAX_STEP - 1) * FEATURE_SIZE).reshape(MAX_STEP - 1,
                                                                 FEATURE_SIZE).tolist()
        with self.fsys.open('raw_data/some00005.csv', mode='w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['data', 'open', 'close', 'max', 'min', 'volumn'])
            for line in lines:
                writer.writerow(line)
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        lines = []
        self.assertFalse(inputdata._raw_data_check('some00005.csv', []))
        # valid the problem not to be comptible logged correctly
        path = inputdata.__default_result_data_dir__ + inputdata.__default_log_file__
        logcontent = []
        with self.fsys.open(path, mode='r') as logerror:
            reader = csv.reader(logerror)
            next(reader)
            logcontent = next(reader)
        self.assertEqual(logcontent[0], 'some00005.csv')
        self.assertEqual(int(logcontent[1]), rnn_model_exception.DataNotComptible.is_not_enough)
    #
    def test_raw_data_check_has_non_number(self):
        lines = np.arange(MAX_STEP * FEATURE_SIZE).reshape(MAX_STEP, FEATURE_SIZE).tolist()
        line = lines[5]
        line[2] = 'awev'
        lines[5] = line
        with self.fsys.open('raw_data/some00005.csv', mode='w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['data', 'open', 'close', 'max', 'min', 'volumn'])
            for _line in lines:
                writer.writerow(_line)
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        lines = []
        self.assertFalse(inputdata._raw_data_check('some00005.csv', lines))
        # valid the problem not to be comptible logged correctly
        path = inputdata.__default_result_data_dir__ + inputdata.__default_log_file__
        logcontent = []
        with self.fsys.open(path, mode='r') as logerror:
            reader = csv.reader(logerror)
            next(reader)
            logcontent = next(reader)
        self.assertEqual(logcontent[0], 'some00005.csv')
        self.assertEqual(int(logcontent[1]), rnn_model_exception.DataNotComptible.has_non_float)
    #
    def test_raw_data_check_result(self):
        _lines = []
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        inputdata._raw_data_check("some00001.csv", _lines)
        self.assertGreater(len(_lines), 0)
    '''def __raw_data_log_assert_method(self, filename, error_type):
        with self.fsys.open('result_data/result_data/dataset' +
                            str(MAX_STEP) + '/logerror.txt', mode='r') as logfile:
            reader = csv.reader(logfile)
            next(reader)
            logcontent = next(reader)
            filename = logcontent[0]
            error_type = int(logcontent[1])
        self.assertEqual(filename, 'some00005.csv')
        self.assertEqual(error_type, rnn_model_exception.DataNotComptible.has_non_float)'''
#
class test_make_examples_for_prediction(tf.test.TestCase):
    def setUp(self):
        self.fsys = get_fsys()
    def tearDown(self):
        self.fsys.close()
    def test_make_single_example(self):
        ''' test the method _make_examples_for_prediction
            of InputData creates tf.train.SequenceExample
        '''
        #preparation
        class InputData_MakeSingleExample(ldp.InputData):
            ''' from _save_example_for_prediction check the indirect output
                of _make_examples_for_prediction
                so the method _save_example_for_prediction will be overridden as check point
            '''
            self.__context_parsed__ = None
            self.__sequence_parsed__ = None
            def _save_example_for_prediction(self, ex):
                ''' parser the ex to valid if it is created correctly
                '''
                ex_serial = ex.SerializeToString()
                context_features = {self.__default_tfcontext_token__:
                                        tf.FixedLenFeature([], dtype=tf.string),
                                    self.__default_tfcontext_sequent_length__:
                                        tf.FixedLenFeature([], dtype=tf.int64)
                                   }
                sequence_features = {self.__default_tfexample_input_sequence__:
                                     tf.FixedLenSequenceFeature([], dtype=tf.float32)
                                    }
                context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                    serialized=ex_serial,
                    context_features=context_features,
                    sequence_features=sequence_features
                    )
                self.__context_parsed__ = context_parsed
                self.__sequence_parsed__ = sequence_parsed

        #
        _lines_array = np.random.normal(size=MAX_STEP*FEATURE_SIZE)
        _lines = _lines_array.reshape(MAX_STEP, FEATURE_SIZE).tolist()
        lines = [[datetime.datetime.now()] + _line for _line in _lines]
        initdata = InputData_MakeSingleExample(self.fsys, MAX_STEP, FEATURE_SIZE)
        #exercise
        initdata._make_examples_for_prediction(lines, 'some00000')
        #valid
        context_token = initdata.__default_tfcontext_token__
        context_length = initdata.__default_tfcontext_sequent_length__
        input_sequence = initdata.__default_tfexample_input_sequence__
        context_parsed = initdata.__context_parsed__
        sequence_parsed = initdata.__sequence_parsed__
        with tf.Session() as sess:
            self.assertEqual(
                sess.run(context_parsed[context_token]),
                b'some00000')
            self.assertEqual(
                sess.run(context_parsed[context_length]),
                MAX_STEP)
            self.assertEquals(
                sess.run(sequence_parsed[input_sequence]),
                np.array(lines))

#

if __name__ == "__main__":
    tf.test.main()
