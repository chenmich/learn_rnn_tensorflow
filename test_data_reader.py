import unittest
import numpy as np
import data_reader as dr
import datetime
import itertools


class test_someData(unittest.TestCase):
    ''' test the some data set
    '''
    def test_data(self):
        ''' test for _train_data and _valid_data
        '''
        #parpara for test
        _num_file = 2
        _length_file = 4000
        _sequence_length = 200
        _files = ['data/some00000.csv', 'data/some00001.csv']
        _some_data = dr.SomeData(_files, _sequence_length)

        #test for _train_data
        _train = np.array(_some_data._train_data)
        datetime0 = datetime.datetime.strptime(_train[0, 0, 0], "%d-%b-%y")
        datetime1 = datetime.datetime.strptime(_train[0, 1, 0], "%d-%b-%y")
        self.assertLess(datetime0.day, datetime1.day)#make sure reveser the order
        self.assertEqual(len(_train[0]), _length_file - _sequence_length)
        #test for _valid_data
        _valid = np.array(_some_data._valid_data)
        datetime0 = datetime.datetime.strptime(_valid[0, 0, 0], "%d-%b-%y")
        datetime1 = datetime.datetime.strptime(_valid[0, 1, 0], "%d-%b-%y")
        self.assertLess(datetime0.day, datetime1.day)#make sure reveser the order
        self.assertEqual(len(_valid[0]), _sequence_length)

#
    def test_train_data_property(self):
        #parpara for test
        _num_file = 1
        _length_file = 4000
        _sequence_length = 200
        _files = ['data/some00000.csv']
        _some_data = dr.SomeData(_files, _sequence_length)
        _data_set = _some_data.train_data()
        n = 0
        




if __name__ == "__main__":
    unittest.main()
    