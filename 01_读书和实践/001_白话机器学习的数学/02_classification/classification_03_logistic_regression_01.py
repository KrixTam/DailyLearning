# 逻辑回归
import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('./data/image2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# 初始化参数
theta = np.random.rand(3)

# 标准化
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# 增加x0
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])


X = to_matrix(train_z)


# sigmoid函数
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# 学习率
ETA = 1e-3

# 重复次数
epoch = 5000

# 重复学习
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

x0 = np.linspace(-2, 2, 100)


def classify(x):
    return (f(x) >= 0.5).astype(np.int)


print(classify(to_matrix(standardize([
    [200, 100],  # 200×100的横向图像，输出为：1
    [100, 200]  # 100×200的横向图像，输出为：0
]))))

# 将标准化后的训练数据画成图
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()
