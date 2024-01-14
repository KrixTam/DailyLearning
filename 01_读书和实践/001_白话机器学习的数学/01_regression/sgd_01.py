# 随机梯度下降法 stochastic gradient descent
import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('./data/click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]
#
# train_x = np.asarray([ 281.23529412, 841.70588235, 1402.17647059, 1962.64705882, 2523.11764706,
#  3083.58823529, 3644.05882353, 4204.52941176, 4765., 5325.47058824,
#  5885.94117647, 6446.41176471, 7006.88235294, 7567.35294118, 8127.82352941,
#  8688.29411765, 9248.76470588])
# train_y = np.asarray([147.,  18.,   7.,   7.,   6.,   8.,   6.,   3.,   4.,   4.,   6.,   4.,   2.,   3.,
#    2.,   2.,   2.])

# 标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# 均方误差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# 初始化参数
theta = np.random.rand(3)
# 均方误差的历史记录
error = []


# 创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


X = to_matrix(train_z)


# 预测函数
def f(x):
    return np.dot(x, theta)


# 目标函数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# 学习率
ETA = 1e-3

# 误差的差值
diff = 1

# 重复学习
error.append(MSE(X, train_y))
while diff > 1e-2:
    # 为了调整训练数据的顺序，准备随机的序列
    p = np.random.permutation(X.shape[0])
    # 随机取出训练数据，使用随机梯度下降法更新参数
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    # 计算与上一次误差的差值
    error.append(MSE(X, train_y))
    diff = error[-2] - error[-1]

# 绘图确认
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
