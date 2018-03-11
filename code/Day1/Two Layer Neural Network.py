"""
此样例输入集X为4*3，有4个样本3个属性，每一个样本对应与一个真实值y，为4*1的向量，
我们要根据input的值输出与y值损失最小的输出。
f(w1*x1 + w2*x2 + w3*x3)
f为sigmoid函数
"""


"""
神经网络的优化过程是：
1. 前向传播求损失
2. 反向传播更新w
"""


import numpy as np

# sigmoid function
# deriv=true 时求的是导数


def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


# input dataset

X = np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 1],
              [0, 1, 1]
              ])

# output dataset

y = np.array([[0, 1, 1, 0]]).T


# seed random numbers to make calculation
np.random.seed(1)


# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1


# 迭代次数
for iter in range(10000):
    # forward propagation
    # l0也就是输入层
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y-l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error*nonlin(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)


print("Output After Training:")
print(l1)
