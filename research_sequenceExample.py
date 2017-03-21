import tensorflow as tf

ex = tf.train.SequenceExample()

ex.context.feature['length_fea'].int64_list.value.append(5)
ex.context.feature['length_seq'].int64_list.value.append(200)

t0 = ex.feature_lists.feature_list["tokens0"]
l0 = ex.feature_lists.feature_list['labels0']
t1 = ex.feature_lists.feature_list['tokens1']
l1 = ex.feature_lists.feature_list['labels1']

t0.feature.add().float_list.value.append(2.0)
l0.feature.add().float_list.value.append(3.0)
t1.feature.add().float_list.value.append(4.0)
l1.feature.add().float_list.value.append(5.0)
t0.feature.add().float_list.value.append(2.0)
l0.feature.add().float_list.value.append(3.0)
t1.feature.add().float_list.value.append(4.0)
l1.feature.add().float_list.value.append(5.0)
t0.feature.add().float_list.value.append(2.0)
l0.feature.add().float_list.value.append(3.0)
t1.feature.add().float_list.value.append(4.0)
l1.feature.add().float_list.value.append(5.0)

context_features = {
    "length_fea": tf.FixedLenFeature([], dtype=tf.int64),
    "length_seq": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens0": tf.FixedLenSequenceFeature([], dtype=tf.float32),
    "labels0": tf.FixedLenSequenceFeature([], dtype=tf.float32), 
    "tokens1": tf.FixedLenSequenceFeature([], dtype=tf.float32),
    "labels1": tf.FixedLenSequenceFeature([], dtype=tf.float32),    
}

context_parsed, sequence_parsed = tf.parse_single_sequence_example(
    serialized=ex.SerializeToString(),
    context_features=context_features,
    sequence_features=sequence_features
)

print()
print('================================================================')

print(context_parsed)
print()
print()
print(sequence_parsed)
print()
print()

with tf.Session() as sess:
    print("length_fea    ", sess.run(context_parsed['length_fea']))
    print("length_seq    ", sess.run(context_parsed["length_seq"]))
    print()
    print("tokens0     ", sess.run(sequence_parsed["tokens0"]))
    print("tokens1     ", sess.run(sequence_parsed["tokens1"]))
    pass



