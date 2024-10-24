import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x>0, dtype=np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # 生成图3-8
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    plt.plot(x, y2)
    plt.plot(x, y1, '--')
    plt.ylim(-0.1, 1.1)  # 限定y轴的显示范围
    plt.show()
