import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('./data/click.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

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
    # 更新参数
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 计算与上一次误差的差值
    error.append(MSE(X, train_y))
    diff = error[-2] - error[-1]

# 绘制误差变化图
x = np.arange(len(error))
plt.plot(x, error)
plt.show()
