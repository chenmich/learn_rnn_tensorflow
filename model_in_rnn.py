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
BATCH_SIZE = 2
MAX_STEP = 10
FEATURE_SIZE = 8
NUM_UNITS = 64
# Create input data
X = np.random.randn(BATCH_SIZE, MAX_STEP, FEATURE_SIZE)
Y = X*np.sin(X)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS, state_is_tuple=True)
state = cell.zero_state(BATCH_SIZE, dtype=tf.float64)
loss = 0
W = tf.get_variable(dtype=tf.float64, shape=[NUM_UNITS, FEATURE_SIZE], name='weight')
B = tf.get_variable(dtype=tf.float64, shape=[FEATURE_SIZE], name='bais')
output, state = tf.nn.dynamic_rnn(cell, X, [MAX_STEP, MAX_STEP], state)
shape = output.shape
print("output'shape is ", shape)
print("Y's shape is ", Y.shape)
for i in range(shape[0]):
    _y = tf.matmul(output[i], W) + B
    loss += tf.reduce_sum(tf.square(Y[i] - _y))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for _ in range(100):
    sess.run(train)
print(sess.run(loss))
