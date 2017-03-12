import tensorflow as tf
import numpy as np


cell = tf.contrib.rnn.BasicLSTMCell(num_units=64, state_is_tuple=True)
state = cell.zero_state(3, dtype=tf.float32)
W = tf.get_variable(name='weigth')
B = tf.get_variable(name='bais')
def lossFn(predicted, real):
    substract_tensor = predicted - real
    substract_square_tensor = tf.square(substract_tensor)
    loss = tf.reduce_sum(substract_square_tensor)
    return loss
def predicted(outputs):
    predict_values = []
    for i in range(3):
        predicted_tensor = tf.matmul(outputs[i], W) + B
        predict_values.append(predicted_tensor)
    prediction = tf.stack([x for x in predict_values])
    return prediction
x = np.zeros((3, 2, 3))
y = x + 2.0
_x = np.reshape(x, (3*2*3))
_y = np.reshape(y, (3*2*3))
print(x)
print(y)
x_tensor = tf.Variable(_x)
y_tensor = tf.Variable(_y)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
Myloss = sess.run(lossFn(x_tensor, y_tensor))
print(Myloss)

