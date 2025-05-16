import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader


# 定义超参数
input_dim = 28 * 28  # MNIST图像展平后的维度
hidden_dim = 128  # 隐藏层神经元数量
output_dim = 10  # 分类类别数
batch_size = 100
num_epochs = 10
learning_rate = 0.001
# learning_rate = 0.01

# 数据加载与预处理
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)