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
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as pyl
import data_reader


#model parameters
BATCH_SIZE = 2
MAX_STEP = 10
FEATURE_SIZE = 8
NUM_UNITS = 64
MU = 1.401155
INIT_VALUE = 0.618
# Create input data
X = np.random.randn(BATCH_SIZE, MAX_STEP, FEATURE_SIZE)
Y = X*np.sin(X)
#model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS, state_is_tuple=True)
state = cell.zero_state(BATCH_SIZE, dtype=tf.float64)
W = tf.get_variable(dtype=tf.float64, shape=[NUM_UNITS, FEATURE_SIZE], name='weight')
B = tf.get_variable(dtype=tf.float64, shape=[FEATURE_SIZE], name='bais')

_X = tf.placeholder(dtype=tf.float64, shape=[BATCH_SIZE, MAX_STEP, FEATURE_SIZE])
output, state = tf.nn.dynamic_rnn(cell, _X, [MAX_STEP, MAX_STEP], state)

loss = 0
for i in range(output.shape[0]):
    _y = tf.matmul(output[i], W) + B
    loss += tf.reduce_sum(tf.square(Y[i] - _y))

#prepare for train
losses = []
echo = []
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#train
for i in range(0, 500, 1):#multi echo of train using same dataset
    sess.run(train, feed_dict={_X: tf.Variable(X)})
    if i % 10 == 0:
        loss_patial = sess.run(loss)
        print("echo= ", i, "loss= ", loss_patial)
        losses.append(loss_patial)
        echo.append(i)
pyl.plot(echo, losses)
pyl.show()
