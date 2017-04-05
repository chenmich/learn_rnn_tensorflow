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

#
MAX_STEP = 200
FEATURE_SIZE = 5
FILE_WILDCARD = '*.csv'
NUM_RAW_FILES = 5
#
#test InputData class
class test_get_files(tf.test.TestCase):
    ''' test the method of InputData' method _get_raw_data_files
    '''
    def test_returned_value(self):
        ''' valid returned value
        '''
        fsys = get_fsys()
        inputdata = ldp.InputData(fsys, MAX_STEP, FEATURE_SIZE)
        pure_path = inputdata.__default_raw_data_dir__
        match = inputdata.__raw_file_wildcard__
        files = inputdata._get_files(pure_path, match)
        self.assertEqual(len(files), NUM_RAW_FILES)
        self.assertEqual(files[0], 'some00000.csv')
        self.assertEqual(files[1], 'some00001.csv')
        self.assertEqual(files[2], 'some00002.csv')
        self.assertEqual(files[3], 'some00003.csv')
        self.assertEqual(files[4], 'some00004.csv')
        fsys.close()

#
class test_setup_result_dir(tf.test.TestCase):

    def test_setup_result_dir(self):
        inputdata = ldp.InputData(get_fsys(), MAX_STEP, FEATURE_SIZE)
        inputdata.__setup_result_dir__()
        fsys = inputdata.__fsys_data__
        pure_path = inputdata.__default_result_data_dir__
        self.assertTrue(fsys.exists(pure_path))
        self.assertTrue(fsys.exists(pure_path + 'logerror.txt'))
        self.assertTrue(fsys.exists(pure_path + 'test' +
                                    inputdata.__bits_of_file_number__*'0' + '.tfrecord'))
        self.assertTrue(fsys.exists(pure_path + 'prediction' +
                                    inputdata.__bits_of_file_number__*'0' + '.tfrecord'))
        self.assertTrue(fsys.exists(pure_path + 'train' +
                                    inputdata.__bits_of_file_number__*'0' + '.tfrecord'))
        self.assertTrue(fsys.exists(pure_path + 'valid' +
                                    inputdata.__bits_of_file_number__*'0' + '.tfrecord'))
        with fsys.open(pure_path + 'logerror.txt', mode='r') as logerror:
            reader = csv.reader(logerror)
            line = next(reader)
            self.assertEqual(line[0], 'file_name')
            self.assertEqual(line[1], 'error_type')

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
            def _raw_data_check(self, filename):
                # force to return a True fot to test if _make_examples will call the method
                lines = np.random.normal(size=10*MAX_STEP*FEATURE_SIZE).reshape(
                    10*MAX_STEP, FEATURE_SIZE).tolist()
                return True, lines
            def _make_examples_for_prediction(self, lines, tokens):
                self._make_examples_for_prediction_called = True
            # separat the effect of this method
            def _make_examples_for_trains(self, lines, tokens):
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
            def _raw_data_check(self, filename):
                # force to return a True fot to test if _make_examples will call the method
                lines = np.random.normal(size=10*MAX_STEP*FEATURE_SIZE).reshape(
                    10*MAX_STEP, FEATURE_SIZE).tolist()
                return True, lines
            def _make_examples_for_prediction(self, lines, tokens):
                pass
            def _make_examples_for_trains(self, lines, tokens):
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
        self.assertTrue(inputdata._raw_data_check('some00000.csv'))
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
        is_comptible, lines = inputdata._raw_data_check('some00005.csv')
        self.assertFalse(is_comptible)
        self.assertLess(len(lines), MAX_STEP)
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
        self.assertFalse(inputdata._raw_data_check('some00005.csv'))
        # valid the problem not to be comptible logged correctly
        log_path = inputdata.__default_result_data_dir__ + inputdata.__default_log_file__
        logcontent = ['some00005.csv', rnn_model_exception.DataNotComptible.has_non_float]
        self.__raw_not_comptible_log_assertion(log_path, logcontent)
    def __raw_not_comptible_log_assertion(self, log_path, logcontent):
        with self.fsys.open(log_path, mode='r') as logerror:
            reader = csv.reader(logerror)
            next(reader)
            _logcontent = next(reader)
        self.assertEqual(_logcontent[0], logcontent[0])
        self.assertEqual(int(_logcontent[1]), logcontent[1])
    def test_raw_data_check_result(self):
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        is_comptible, lines = inputdata._raw_data_check("some00001.csv")
        self.assertGreater(len(lines), 0)
        self.assertTrue(is_comptible)

#
class test_encode_decode_example_prediction(tf.test.TestCase):
    def setUp(self):
        self.fsys = get_fsys()
    def tearDown(self):
        self.fsys.close()
    def test_encode_decode_example_prediction(self):
        ''' test the method _make_examples_for_prediction
            of InputData creates tf.train.SequenceExample
        '''
        #preparation
        #
        _lines_array = np.random.normal(size=MAX_STEP*FEATURE_SIZE)
        _lines = _lines_array.reshape(MAX_STEP, FEATURE_SIZE).tolist()
        #simulate the raw data
        lines = [[str(datetime.date.today())] + _line for _line in _lines]
        start_expect = bytes(lines[MAX_STEP - 1][0], 'utf-8')
        end_expect = bytes(lines[0][0], 'utf-8')
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        context_token = inputdata.__default_token__
        context_length = inputdata.__default_sequent_length__
        context_start = inputdata.__input_sequence_start_date__
        context_end = inputdata.__input_sequence_end_date__
        input_sequence = inputdata.__example_input_sequence__

        #exercise
        ex = inputdata._encode_prediction_example(lines, 'some00000')

        #valid
        context_parsed, sequence_parsed = inputdata._decode_prediction_example(
            ex.SerializeToString())

        with tf.Session() as sess:
            self.assertEqual(
                sess.run(context_parsed[context_token]),
                b'some00000')
            self.assertEqual(
                sess.run(context_parsed[context_length]),
                MAX_STEP)
            self.assertEqual(
                sess.run(context_parsed[context_start]),
                start_expect
            )
            self.assertEqual(
                sess.run(context_parsed[context_end]),
                end_expect
            )
            shape = sequence_parsed[input_sequence].get_shape()
            real = sess.run(sequence_parsed[input_sequence])
            expect = np.array(_lines).flatten()
            self.assertAllClose(real, expect)
    #

#
class test_encode_decode_example_trains(tf.test.TestCase):
    def setUp(self):
        self.fsys = get_fsys()
    def tearDown(self):
        self.fsys.close()
    def test_encode_decode(self):
        ''' This test will encode lines of train example to tf.SequenceExample and decode it
            And valid the result
        '''
        #prepare a example for trains
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        max_step = inputdata.__max_step__
        feature_size = inputdata.__feature_size__
        _token = 'some00000'
        _lines = np.random.normal(size=2*max_step*feature_size).reshape(
            2*max_step, feature_size).tolist()
        lines = [[str(datetime.date.today())] + _line for _line in _lines]
        lines.reverse()
        input_sequence = lines[0:MAX_STEP]
        target_sequence = lines[MAX_STEP:]
        example_line = {inputdata.__example_input_sequence__: input_sequence,
                        inputdata.__example_target_sequence__: target_sequence}
        #exercise
        ex = inputdata._encode_train_example(example_line, _token)
        context_parsed, sequnece_parsed = inputdata._decode_train_example(ex.SerializeToString())

        #prepare expect content
        input_start = bytes(lines[0][0], 'utf-8')
        input_end = bytes(lines[max_step - 1][0], 'utf-8')
        target_start = bytes(lines[max_step][0], 'utf-8')
        target_end = bytes(lines[2*max_step - 1][0], 'utf-8')
        token = bytes(_token, 'utf-8')
        length = len(_lines) // 2
        _inputs = [line[1:] for line in input_sequence]
        _target = [line[1:] for line in target_sequence]
        input_sequence = np.array(_inputs).flatten()
        target_sequence = np.array(_target).flatten()

        #valid
        with tf.Session() as sess:
            self.assertEqual(sess.run(context_parsed[inputdata.__default_sequent_length__]),
                             length)
            self.assertEqual(sess.run(context_parsed[inputdata.__default_token__]),
                             token)
            self.assertEqual(sess.run(context_parsed[inputdata.__input_sequence_start_date__]),
                             input_start)
            self.assertEqual(sess.run(context_parsed[inputdata.__input_sequence_end_date__]),
                             input_end)
            self.assertEqual(sess.run(context_parsed[inputdata.__target_sequence_start_date__]),
                             target_start)
            self.assertEqual(sess.run(context_parsed[inputdata.__target_sequence_end_date__]),
                             target_end)
            self.assertAllClose(sess.run(sequnece_parsed[inputdata.__example_input_sequence__]),
                                input_sequence)
            self.assertAllClose(sess.run(sequnece_parsed[inputdata.__example_target_sequence__]),
                                target_sequence)


#
class test_make_example_for_trains(tf.test.TestCase):
    ''' this test is for make_for_trains
        Firstly, I will write this test class to complete the test
        for make_example_for_trains by from top to down
    '''
    def setUp(self):
        self.fsys = get_fsys()
    def tearDown(self):
        self.fsys.close()
    def test_divide_line(self):
        ''' This test checks whether the divided lines of examples are correct
            asssume a short example which time step less than specified max_step
            the number of raw data is 3800,and max_step is 200
            There will have  max_step lines of input sequence and a max_step of target sequence
            in a completed example
        '''
        token = 'some00000'
        #prepare the lines of raw data
        num_example = 10
        #there are nime full example and one short example
        num_line = (num_example -1)*(2*MAX_STEP) + MAX_STEP
        _lines = np.random.normal(loc=0.0, scale=1.0,
                                  size=num_line*FEATURE_SIZE).reshape(num_line,
                                                                      FEATURE_SIZE).tolist()
        #keep the data of date of exchange
        lines = [[str(datetime.date.today())] + _line for _line in _lines]
        inputdata = ldp.InputData(self.fsys, MAX_STEP, FEATURE_SIZE)
        #exercise
        example_lines = inputdata._divide_line(lines)
        #at this time, I can determine the structur of example_lines
        self.assertEqual(num_example, len(example_lines))
        self.assertEqual(MAX_STEP, len(example_lines[0]['input_sequence']))
        self.assertEqual(MAX_STEP, len(example_lines[0]['target_sequence']))
        #the earlier of data the less effect for future price
        self.assertEqual(MAX_STEP // 2, len(example_lines[num_example - 1]['input_sequence']))
        self.assertEqual(MAX_STEP // 2, len(example_lines[num_example - 1]['target_sequence']))
    #
    def test_calculate_statistical_feature(self):
        # the already available stat feature of examples for train will be stored 
        # in the field of InputData
        # If there is any added, the stat feature will be calculated by the function
        # in the method calculate_statistical_feature 
        
        def Assertion(expect, real):
            self.assertEqual(expect.num, real.num)
            self.assertAllClose(expect.mean, real.mean)
            self.assertAllClose(expect.std, real.std)

        def get_example(lines):
            _lines = [[str(datetime.date.today())] + line for line in lines.tolist()]       
            _lines.reverse()
            example = {'input_sequence': _lines[0:MAX_STEP],
                       'target_sequence': _lines[MAX_STEP:]}
            return example
        def get_stat(lines):
            shape = np.shape(lines)
            stat = ldp.stat_feature()
            stat.num = shape[0]*shape[1]
            stat.mean = np.mean(lines)
            stat.std = np.std(lines)
            return stat
        
        
        #prepare data
        lines = np.random.normal(size=2*MAX_STEP*
                                                FEATURE_SIZE).reshape(2*MAX_STEP, FEATURE_SIZE)
        anotherLines = np.random.normal(size=2*MAX_STEP*
                                                       FEATURE_SIZE).reshape(2*MAX_STEP, FEATURE_SIZE)
        
        example = get_example(lines)
        anotherExample = get_example(anotherLines)
        stat_price = get_stat(lines[0:, 0:FEATURE_SIZE - 1])
        stat_volumn = get_stat(lines[0:, FEATURE_SIZE - 1:])
        
        combinat_lines = np.array(lines.tolist() + anotherLines.tolist())
        combinat_stat_price = get_stat(combinat_lines[0:, 0: FEATURE_SIZE - 1])
        combinat_stat_volumn = get_stat(combinat_lines[0:, FEATURE_SIZE - 1:])

        #exercise
        inputdata = ldp.InputData(get_fsys(), MAX_STEP, FEATURE_SIZE)
        inputdata._calculate_statistical_feature(example)
        #valid        
        Assertion(stat_price, inputdata._stat_features[inputdata.__stat_price__])
        Assertion(stat_volumn, inputdata._stat_features[inputdata.__stat_volumn__])
        #another exercise and valid
        inputdata._calculate_statistical_feature(anotherExample)
        Assertion(combinat_stat_price, inputdata._stat_features[inputdata.__stat_price__])
        Assertion(combinat_stat_volumn, inputdata._stat_features[inputdata.__stat_volumn__])
    #
#
class test_save_examples(tf.test.TestCase):
    pass
#
if __name__ == "__main__":
    tf.test.main()
