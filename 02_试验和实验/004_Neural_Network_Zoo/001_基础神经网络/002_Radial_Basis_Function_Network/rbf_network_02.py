# 比较不同的聚类中心数的效果
# 结合auto_select_k.py，可以发现，自动选择聚类中心数，对RBFNetwork没有太大作用，我们并没有使用最优聚类中心数，反而是节点数越多，效果越好。
# Step 1: Import required libraries

# import required libraries
import torch
from torch import nn, optim
from sklearn.cluster import KMeans
from rbf_show_chart import chart_compared

# Step 2: Load MNIST dataset and data loader

from rbf_data import train_loader, test_loader, train_dataset, learning_rate, input_dim, num_epochs

# Step 4: Define the RBF network

from rbf_network import RBFNetwork


# Step 6: Define method for training

# define training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, input_dim).to(device), target.to(device)
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Loss and accuracy
        running_loss += loss.item()
    return running_loss / len(train_loader)


# Step 7: Define method for testing

# define testing function
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.view(-1, input_dim).to(device), target.to(device)
            # Forward pass
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total


# Step 8: Perform training and testing and evaluate the result

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    results = {}

    # Step 3: Transform the training dataset to fit into K-Means

    X_train = train_dataset.data.reshape(-1, 784).double().to(device) / 255.0

    n_clusters = [10, 20, 30, 50, 100]

    for num_clusters in n_clusters:

        # Cluster the data using KMeans with k=10
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X_train.cpu().numpy())

        # find the cluster centers
        clusters = kmeans.cluster_centers_.astype(float)
        # print(clusters.shape)

        # Step 5: Define Model, optimizer and loss function

        model = RBFNetwork(clusters).to(device)

        # criteria function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        test_name = 'Cluster=' + str(num_clusters)
        print(f"\nTraining {test_name} Model...")

        train_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            test_acc = test_model(model, test_loader, device)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

        results[test_name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies
        }

    chart_compared(results)