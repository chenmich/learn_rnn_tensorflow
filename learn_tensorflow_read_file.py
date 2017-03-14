import tensorflow as tf
filename_queue = tf.train.string_input_producer(["some.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [['1'], [1.0], [1.0], [1.0], [1.0], [1.0]]
col1, col2, col3, col4, col5, col6 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col2, col3, col4, col5, col6])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col1])

  coord.request_stop()
  coord.join(threads)