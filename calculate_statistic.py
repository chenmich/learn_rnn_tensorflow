''' This module will calculate the statistic value when the number of data, mean and
    std of two batch of data
    the equations are:
    e = a1*mean1 + a2*mean2
    sigma^2 = a1*(sigma1^2 + e1^2) + a2*(sigma2^2 + e2^2) - e*e
    a1 = n1 / (n1 + n2)
    a2 = n2 / (n1 + n2)
'''

import numpy as np
valid_data_batch1 = [1, 2, 3, 4, 5]
valid_data_batch2 = [7, 8, 9, 10]
n1 = len(valid_data_batch1)
n2 = len(valid_data_batch2)

a1 = n1/(n1 + n2)
a2 = n2/(n1 + n2)

e1 = np.mean(valid_data_batch1)
e2 = np.mean(valid_data_batch2)
sigma_square1 = np.square(np.std(valid_data_batch1))
sigma_square2 = np.square(np.std(valid_data_batch2))


e = a1*e1 +a2*e2
sigma_square = a1*(sigma_square1 + e1*e1) + a2*(sigma_square2 + e2*e2) - e*e


expect_e = np.mean(valid_data_batch1 + valid_data_batch2)
expect_sigma_square = np.square(np.std(valid_data_batch1 + valid_data_batch2))

print ("e", e, expect_e)
print("sigma_square", sigma_square, expect_sigma_square)