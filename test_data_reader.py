import unittest
import numpy as np
import data_reader as dr
import datetime


class test_someData(unittest.TestCase):
    ''' test the some data set
    '''
    def test_train_data(self):
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
        self.assertEqual(_train.shape[1], _length_file - _sequence_length)
        #test for _valid_data
        _valid = np.array(_some_data._valid_data)
        datetime0 = datetime.datetime.strptime(_valid[0, 0, 0], "%d-%b-%y")
        datetime1 = datetime.datetime.strptime(_valid[0, 1, 0], "%d-%b-%y")
        self.assertLess(datetime0.day, datetime1.day)#make sure reveser the order
        self.assertEqual(_valid.shape[1], _sequence_length)





if __name__ == "__main__":
    unittest.main()
    