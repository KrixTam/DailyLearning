import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
training_ds = np.loadtxt('./data/click.csv', delimiter=',', skiprows=1)
train_x = training_ds[:, 0]
train_y = training_ds[:, 1]

# 绘图
# plt.plot(train_x, train_y, 'o')

theta_0 = np.random.rand()
theta_1 = np.random.rand()


def f(x):  # 预测函数
    return theta_0 + theta_1 * x


def E(x, y):  # 目标函数
    return 0.5 * np.sum((y - f(x)) ** 2)


def standardize(x: np.ndarray):  # 标准化/z-score规范化
    mu = x.mean()
    sigma = x.std()
    return (x - mu) / sigma


train_z = standardize(train_x)

# 学习率
ETA = 1e-3
# 误差的差值
diff = 1
# 更新次数
count = 0

# 重复学习（训练）
error = E(train_z, train_y)
while diff > 1e-2:
    # 更新结果保存到临时变量
    tmp_0 = theta_0 - ETA * np.sum(f(train_z) - train_y)
    tmp_1 = theta_1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    # 更新参数
    theta_0 = tmp_0
    theta_1 = tmp_1
    # 计算与上一次误差的差值
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    # 输出日志
    count += 1
    log = '第{}次：theta_0 = {:.3f}，theta_1 = {:.3f}，差值 = {:.4f}'
    print(log.format(count, theta_0, theta_1, diff))

# 绘图
plt.plot(train_z, train_y, 'o')

# 绘制结果
x = np.linspace(-3, 3, 100)
plt.plot(x, f(x))

plt.show()
