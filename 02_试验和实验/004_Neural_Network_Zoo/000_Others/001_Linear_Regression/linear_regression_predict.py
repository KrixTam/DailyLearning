import torch
import matplotlib.pyplot as plt
from linear_regression_data import x_train, y_train
from linear_regression_train_cpu import LinearRegressionModel
# from linear_regression_train_gpu import LinearRegressionModel


input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim, output_dim)


# load_model = False
load_model = True
if load_model is True:
    model.load_state_dict(torch.load('linear_regression_model.pkl'))

# Clear figure
plt.clf()

# Get predictions
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

# Plot true data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()