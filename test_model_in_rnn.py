import tensorflow as tf
import numpy as np
a = np.random.randn(3,5,8)
b = np.reshape(a, 3*5*8)
tensor_b = tf.Variable(b)
tensor_a = tf.reshape(tensor_b, shape=[3, 5, 8])
tensor_c = tf.reshape(tensor_b, shape=[3, 5, 8])
tensor_d = tensor_a - tensor_c
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(tensor_d))
