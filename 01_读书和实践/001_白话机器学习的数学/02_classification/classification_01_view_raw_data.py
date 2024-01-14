import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt('./data/image1.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# 绘图
plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.axis('scaled')
plt.show()
