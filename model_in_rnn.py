# Copyright 2017 The Chenmich Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
''' This modular is for model
'''
import datetime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as pyl
import data_reader as dr


#model parameters
BATCH_SIZE = 5
NUM_BATCH = 10000
SEQUENCE_LENGTH = 200
FEATURE_SIZE = 5
MU = 1.401157
INIT_VALUE = 0.618

NUM_UNITS = 128
NUM_CELL_STACK = 3
ECHO = 5000
def printFn(content):
    for _ in range(10):
        print(content)
sess = tf.Session()
#model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS, state_is_tuple=True)
cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(NUM_CELL_STACK)])
state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)

W = tf.get_variable(name="weight", shape=[NUM_UNITS, FEATURE_SIZE], dtype=tf.float64)
B = tf.get_variable(name="bias", shape=[SEQUENCE_LENGTH, FEATURE_SIZE],dtype=tf.float64)
_X = tf.placeholder(dtype=tf.float64, shape=[BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_SIZE])
_Y = tf.placeholder(dtype=tf.float64, shape=[BATCH_SIZE, SEQUENCE_LENGTH, FEATURE_SIZE])
outputs, state = tf.nn.dynamic_rnn(cell=cells, inputs=_X,
                                   sequence_length=[SEQUENCE_LENGTH]*BATCH_SIZE,
                                   dtype=tf.float64)
#computer predicted value
def predicted(outputs):
    predict_values = []
    for i in range(BATCH_SIZE):
        predicted_tensor = tf.matmul(outputs[i], W) + B
        predict_values.append(predicted_tensor)
    prediction = tf.stack([x for x in predict_values])
    return prediction

def lossFn(predicted, real):
    substract_tensor = predicted - real
    substract_square_tensor = tf.square(substract_tensor)
    loss = tf.reduce_sum(substract_square_tensor) / BATCH_SIZE
    return loss
#
loss = lossFn(predicted(outputs), _Y)
#prepare for train
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
#train
print()
print()
print()
print("Training the model......")
pyl_x = []
pyl_y = []
start = datetime.datetime.now()
print("start time is ", start)
for echo in range(0,ECHO+1):
    losses = []
    
    for x, y in dr.data_reader(num_batch=NUM_BATCH, batch_size=BATCH_SIZE,
                               sequence_length=SEQUENCE_LENGTH,
                               feature_size=FEATURE_SIZE):
        sess.run(train, feed_dict={_X:x, _Y:y})
        _loss = sess.run(loss, feed_dict={_X:x, _Y:y})
        losses.append(_loss)
    _loss_ = np.sum(losses, 0) / NUM_BATCH
    if (echo) % 10 == 0:
        print(echo, _loss_)
        pyl_x.append(echo)
        pyl_y.append(_loss_)

end = datetime.datetime.now()
print()
print("The training is over")
print("The end time is ", end)
print("total time for training is ", end-start)

pyl.plot(pyl_x, pyl_y)
pyl.show()