import numpy as np
import matplotlib.pyplot as plt


# 真正的数据
def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)


# 随意准备一些向真正的函数加入了一点噪声的训练数据
train_x = np.linspace(-1.5, 1.5, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

# 标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# 创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([
        np.ones(x.size),
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        x ** 5,
        x ** 6,
        x ** 7,
        x ** 8,
        x ** 9,
        x ** 10,
    ]).T


X = to_matrix(train_z)

# 参数初始化
theta = np.random.randn(X.shape[1])


# 预测函数
def f(x):
    return np.dot(x, theta)


# 目标函数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# 学习率
ETA = 1e-4

# 误差
diff = 1

# 重复学习
error = E(X, train_y)
while diff > 1e-6:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# 绘图确认
x = np.linspace(-1.5, 1.5, 100)
plt.plot(train_x, train_y, 'o')
plt.plot(x, g(x), linestyle='dashed')
plt.ylim(-1, 2)
# 对学习结果绘图
z = standardize(x)
plt.plot(z, f(to_matrix(z)))
plt.show()
