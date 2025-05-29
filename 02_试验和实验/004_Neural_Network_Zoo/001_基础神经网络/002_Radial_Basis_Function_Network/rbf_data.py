import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader


# 定义超参数
input_dim = 28 * 28  # MNIST图像展平后的维度
learning_rate = 0.01
num_epochs = 10

# Load MNIST dataset
train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
