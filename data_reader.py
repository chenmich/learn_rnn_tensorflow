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
''' Generate a serie of parabolic mapping
'''
import numpy as np
from matplotlib import pyplot as pyplot



def data_reader(num_batch=1000, batch_size=5,
                sequence_length=200, feature_size=5,
                mu=1.401155189, init_value=0.618):
    ''' generate a na.array object wiht shape = [batch_size, sequence_length, feature_size]
        Arg:
            num_batch:number of batch about data_reader
            batch_size:quantity of batches
            sequence_length: length of series
            feature_size:feature number of one data_reader
            mu:parameter of parabolic mapping
            init_value:initial value for mapping
    '''
    for _ in range(batch_size):
        batch_data = __generate_data__(2*batch_size*sequence_length*feature_size,
                                       mu=mu, init_value=init_value)#one dimension data
        #cut up to two series
        batch_data_x = batch_data[0: batch_size*sequence_length*feature_size]
        batch_data_y = batch_data[batch_size*sequence_length*feature_size:]
        #reshape to (batch_size, sequence_length, feature_size) from batch_size*sequence_length*feature_size
        batch_data_x = np.reshape(batch_data_x, (batch_size, sequence_length, feature_size))
        batch_data_y = np.reshape(batch_data_y, (batch_size, sequence_length, feature_size))
        yield batch_data_x, batch_data_y



#
def __generate_data__(num, mu=1.401155189, init_value=0.618):
    x = init_value
    #set the first 200 data aside
    for _ in range(200):
        x = 1 - mu*x*x
    #generate all the data
    X = []
    for _ in range(num):
        x = 1 - mu*x*x
        X.append(x)
    return np.array(X)
#

