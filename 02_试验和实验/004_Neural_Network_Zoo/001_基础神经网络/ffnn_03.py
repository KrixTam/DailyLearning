# 比较SGD和Adam两个优化器的效果
import torch
import torch.nn as nn
from ffnn_data import input_dim, hidden_dim, output_dim, num_epochs, learning_rate
from ffnn_data import train_loader, test_loader
from ffnn_02 import FFNNReLU
from ffnn_02 import train_model as train, test_model as test
from ffnn_show_chart import chart


if __name__ == '__main__':
    results = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_01 = FFNNReLU(input_dim, hidden_dim, output_dim).to(device)
    model_02 = FFNNReLU(input_dim, hidden_dim, output_dim).to(device)
    model_03 = FFNNReLU(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_sgd_01 = torch.optim.SGD(model_01.parameters(), lr=learning_rate)
    optimizer_sgd_02 = torch.optim.SGD(
        model_02.parameters(),
        lr=learning_rate,
        momentum=0.9,  # 添加动量
        weight_decay=1e-4  # L2正则化
    )
    optimizer_adam = torch.optim.Adam(model_03.parameters(), lr=learning_rate)

    models = {
        'SGD_01': model_01,
        'SGD_02': model_02,
        'Adam': model_03
    }

    for optimizer_name, optimizer in zip(['SGD_01', 'SGD_02', 'Adam'],
                                         [optimizer_sgd_01, optimizer_sgd_02, optimizer_adam]):
        run_model = models[optimizer_name]
        print(f"\nTraining {optimizer_name} Model...")
        train_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            train_loss = train(run_model, train_loader, criterion, optimizer, device)
            test_acc = test(run_model, test_loader, device)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

        results[optimizer_name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }
    chart(results)
