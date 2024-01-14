# 逻辑回归
# 线性不可分分类
import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('./data/data3.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# 初始化参数
theta = np.random.rand(4)

# 精度的历史记录
accuracies = []

# 标准化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# 增加x0和x3
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])


X = to_matrix(train_z)


# sigmoid函数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 分类函数
def classify(x):
    return (f(x) >= 0.5).astype(np.int)


# 学习率
ETA = 1e-3

# 重复次数
epoch = 5000

# 更新次数
count = 0

# 重复学习
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 计算现在的精度
    result = classify(X) == train_y
    accuracy = len(result[result == True]) / len(result)
    accuracies.append(accuracy)
    # 日志输出
    count += 1
    print('第{}次：theta = {}，accuracy = {}'.format(count, theta, accuracy))

# 绘图确认
plt.subplot(1, 2, 1)
x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
# 将精度画成图
plt.subplot(1, 2, 2)
x = np.arange(len(accuracies))
plt.plot(x, accuracies)

plt.show()
