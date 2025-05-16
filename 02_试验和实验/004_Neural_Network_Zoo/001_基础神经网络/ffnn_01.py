import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from ffnn_show_chart import chart

'''
Step 1: Loading MNIST Train Dataset
'''

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

'''
Step 2: Make Dataset Iterable
'''

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


'''
Step 3: Create Model Class
'''


class FeedforwardNeuralNetModelOneHiddenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, non_linearity):
        super(FeedforwardNeuralNetModelOneHiddenLayer, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity
        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.non_linearity = non_linearity
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)
        # Non-linearity  # NON-LINEAR
        # out = self.sigmoid(out)
        # out = self.tanh(out)
        # out = self.relu(out)
        out = self.non_linearity(out)
        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


class FeedforwardNeuralNetModelTwoHiddenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModelTwoHiddenLayer, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 100 --> 10
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)
        return out


class FeedforwardNeuralNetModelThreeHiddenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModelThreeHiddenLayer, self).__init__()
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 2
        out = self.fc3(out)
        # Non-linearity 2
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out


'''
Step 4: Instantiate Model Class
'''

input_dim = 28*28
hidden_dim = 100
output_dim = 10

model_sigmoid = FeedforwardNeuralNetModelOneHiddenLayer(input_dim, hidden_dim, output_dim, nn.Sigmoid())
model_tanh = FeedforwardNeuralNetModelOneHiddenLayer(input_dim, hidden_dim, output_dim, nn.Tanh())
model_relu = FeedforwardNeuralNetModelOneHiddenLayer(input_dim, hidden_dim, output_dim, nn.ReLU())
model_2hy = FeedforwardNeuralNetModelTwoHiddenLayer(input_dim, hidden_dim, output_dim)
model_3hy = FeedforwardNeuralNetModelThreeHiddenLayer(input_dim, hidden_dim, output_dim)

'''
Step 5: Instantiate Loss Class
'''

criterion = nn.CrossEntropyLoss()

'''
Step 6: Instantiate Optimizer Class
'''

learning_rate = 0.1

optimizer_sigmoid = torch.optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
optimizer_tanh = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
optimizer_relu = torch.optim.SGD(model_relu.parameters(), lr=learning_rate)
optimizer_2hy = torch.optim.SGD(model_2hy.parameters(), lr=learning_rate)
optimizer_3hy = torch.optim.SGD(model_3hy.parameters(), lr=learning_rate)

'''
Step 7: Train Model
'''

models = {
    'one_hidden_layer_sigmoid': model_sigmoid,
    'one_hidden_layer_tanh': model_tanh,
    'one_hidden_layer_relu': model_relu,
    'two_hidden_layer_relu': model_2hy,
    'three_hidden_layer_relu': model_3hy
}

results = {}

for optimizer_name, optimizer in zip(['one_hidden_layer_sigmoid', 'one_hidden_layer_tanh', 'one_hidden_layer_relu', 'two_hidden_layer_relu', 'three_hidden_layer_relu'],
                                     [optimizer_sigmoid, optimizer_tanh, optimizer_relu, optimizer_2hy, optimizer_3hy]):
    run_model = models[optimizer_name]
    print(f"\nTraining {optimizer_name} Model...")
    train_losses = []
    test_accuracies = []
    iter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Load images with gradient accumulation capabilities
            images = images.view(-1, 28*28).requires_grad_()

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = run_model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            running_loss += loss.item()

            iter += 1

            if iter % 500 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0

                # Iterate through test dataset
                for images, labels in test_loader:
                    # Load images with gradient accumulation capabilities
                    images = images.view(-1, 28*28).requires_grad_()

                    # Forward pass only to get logits/output
                    outputs = run_model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)

                    # Total correct predictions
                    correct += (predicted == labels).sum()

                accuracy = 100 * correct / total

                test_accuracies.append(accuracy)
                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
        train_losses.append(running_loss / len(train_loader))

    results[optimizer_name] = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }

chart(results)