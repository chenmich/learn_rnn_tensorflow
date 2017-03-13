import unittest
import numpy as np
import data_reader as dr

BATCH_SIZE = 5
SEQUENCE_LENGTH = 200
FEATURE_SIZE = 5

class test_data_reader(unittest.TestCase):
         
    def test_data_shape(self):
        for x, y in dr.non_linear_parabolic_curve_map_data_reader (1000, batch_size=BATCH_SIZE,
                                                                   sequence_length=SEQUENCE_LENGTH,
                                                                   feature_size=FEATURE_SIZE):
            self.assertEqual(x.shape, (BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_SIZE))

class test_csv_reader(unittest.TestCase):
    def test_csv_reader(self):
        for line in dr.csv_data_reader():
            print(line)

if __name__ == "__main__":
    unittest.main()



