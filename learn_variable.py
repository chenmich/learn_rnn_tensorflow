import tensorflow as tf
import data_reader


a = tf.placeholder(dtype=tf.float64)
b = tf.placeholder(dtype=tf.float64)
add_node = a + b
sess = tf.Session()
for x, y in data_reader.data_reader():
    X = tf.Variable(x, dtype=tf.float64)
    Y = tf.Variable(y, dtype=tf.float64)
    addition = sess.run(add_node, feed_dict={a: X, b: Y})
    print(addition)

