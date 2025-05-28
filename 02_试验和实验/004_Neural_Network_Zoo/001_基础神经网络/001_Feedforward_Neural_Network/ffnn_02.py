# 比较ReLU、Tanh、Sigmoid三种激活函数
import torch
import torch.nn as nn
from ffnn_data import input_dim, hidden_dim, output_dim, num_epochs, learning_rate
from ffnn_data import train_loader, test_loader
from ffnn_show_chart import chart

# 定义不同激活函数的模型
class FFNNReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FFNNsigmoid(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class FFNNtanh(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x


# 训练函数（通用）
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, input_dim).to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


# 测试函数（通用）
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, input_dim).to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total


if __name__ == '__main__':
    # 主程序
    devices = ['cpu']  # 如需GPU，改为 ['cuda' if torch.cuda.is_available() else 'cpu']
    results = {}

    for act_name, model_class in zip(['ReLU', 'Sigmoid', 'Tanh'],
                                     [FFNNReLU, FFNNsigmoid, FFNNtanh]):
        model = model_class(input_dim, hidden_dim, output_dim).to(devices[0])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print(f"\nTraining {act_name} Model...")
        train_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, devices[0])
            test_acc = test_model(model, test_loader, devices[0])
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

        results[act_name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }

    # 结果可视化
    chart(results)