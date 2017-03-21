import tensorflow as tf
import numpy as np
import tempfile

tmp_filename = 'tf.tmp'

sequences = [[1, 2, 3], [1, 2], [3, 2, 1]]
label_sequences = [[0, 1, 0], [1, 0], [1, 1, 1]]

def make_example(input_sequence, output_sequence):
    """
    Makes a single example from Python lists that follows the
    format of tf.train.SequenceExample.
    """

    example_sequence = tf.train.SequenceExample()

    # 3D length
    sequence_length = len(input_sequence)

    example_sequence.context.feature["length"].int64_list.value.append(sequence_length)

    input_characters = example_sequence.feature_lists.feature_list["input_characters"]
    output_characters = example_sequence.feature_lists.feature_list["output_characters"]

    for input_character, output_character in zip(input_sequence,
                                                 output_sequence):

        if input_sequence is not None:
            input_characters.feature.add().int64_list.value.append(input_character)

        if output_characters is not None:
            output_characters.feature.add().int64_list.value.append(output_character)

    return example_sequence

# Write all examples into a TFRecords file
def save_tf(filename):
    with open(filename, 'w') as fp:
        writer = tf.python_io.TFRecordWriter(fp.name)
        for sequence, label_sequence in zip(sequences, label_sequences):
            ex = make_example(sequence, label_sequence)
            writer.write(ex.SerializeToString())
        writer.close()

def read_and_decode_single_example(filename):

    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
        }

    sequence_features = {
        "input_characters": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "output_characters": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }


    return serialized_example, context_features, sequence_features

save_tf(tmp_filename)
length_sequence = len(sequences)

ex, context_features,sequence_features = read_and_decode_single_example(tmp_filename)
context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex,
    context_features=context_features,
    sequence_features=sequence_features
)
    