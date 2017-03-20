# from http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
#===================================================================================================
''' try these code
'''

import tensorflow as tf
import numpy as np

# Example with tf.train.batch dynamic padding
# ==================================================

tf.reset_default_graph()

# Create a tensor [0, 1, 2, 3, 4 ,...]
x = tf.range(1, 10, name="x")

# A queue that outputs 0,1,2,3,...
range_q = tf.train.range_input_producer(limit=5, shuffle=False)
slice_end = range_q.dequeue()

# Slice x to variable length, i.e. [0], [0, 1], [0, 1, 2], ....
y = tf.slice(x, [0], [slice_end], name="y")
y_list = [[[1,2,3],[2,3,4],[3,4,5],[1,2,3],[4,5,6]],
          [[4,5,6],[5,6,7]],
          [[6,7,8]],
          [[7,8,9],[8,9,10]],
          [[9,10,11],[10,11,12],[11,12,13]]]
#y = tf.Variable(y_list)


print(y)

# Batch the variable length tensor with dynamic padding
batched_data = tf.train.batch(
    tensors=[y],
    batch_size=5,
    dynamic_pad=True,
    name="y_batch"
)

print(batched_data)
print()
print('_____________________________________________________')

# Run the graph
# tf.contrib.learn takes care of starting the queues for us
res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)
# Print the result
print("Batch shape: {}".format(res[0]["y"].shape))
print(res[0]["y"])
