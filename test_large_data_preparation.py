import unittest
import csv
import large_data_preparation as ldp

class test_make_example(unittest.TestCase):
    def test_examples(self):
        filename = 'data/raw_data/some00000.csv'
        with open(filename, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            lines = []
            for line in reader:
                lines.append(line)
        SEQUENCE_LENGTH = 200
        length = len(lines)
        examples = ldp._make_examples(200, lines[SEQUENCE_LENGTH:])
        _num_example = len(lines[SEQUENCE_LENGTH:]) // (2*SEQUENCE_LENGTH) + 1
        length = len(examples)
        self.assertEqual(length, 2*_num_example)
        _tmp = examples[2*(_num_example - 1)]
        self.assertEqual(len(_tmp), SEQUENCE_LENGTH // 2)
        self.assertEqual(len(examples[2*_num_example - 1]), SEQUENCE_LENGTH //2)
#
if __name__ == "__main__":
    unittest.main()