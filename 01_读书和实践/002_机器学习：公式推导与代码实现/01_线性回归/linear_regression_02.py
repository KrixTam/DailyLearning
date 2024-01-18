# 导入线性回归模块
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# ===== 导入数据 =====
# 导入load_diabetes模块
from sklearn.datasets import load_diabetes
# 导入打乱数据函数
from sklearn.utils import shuffle
# 获取diabetes数据集
diabetes = load_diabetes()
# 获取输入和标签
data, target = diabetes.data, diabetes.target
# 打乱数据集
X, y = shuffle(data, target, random_state=13)
# 按照8∶2划分训练集和测试集
offset = int(X.shape[0] * 0.8)
# 训练集
X_train, y_train = X[:offset], y[:offset]
# 测试集
X_test, y_test = X[offset:], y[offset:]
# 将训练集改为列向量的形式
y_train = y_train.reshape((-1,1))
# 将测试集改为列向量的形式
y_test = y_test.reshape((-1,1))
# 打印训练集和测试集的维度
print("X_train's shape: ", X_train.shape)
print("X_test's shape: ", X_test.shape)
print("y_train's shape: ", y_train.shape)
print("y_test's shape: ", y_test.shape)


# 定义模型实例
regr = linear_model.LinearRegression()
# 模型拟合训练数据
regr.fit(X_train, y_train)
# 模型预测值
y_pred = regr.predict(X_test)
# 输出模型均方误差
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# 计算R2系数
print('R Square score: %.2f' % r2_score(y_test, y_pred))
