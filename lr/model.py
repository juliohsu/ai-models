import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# create random dataset
torch.manual_seed(42)
X = torch.linspace(0, 10, 100).reshape(-1, 1)
Y = X * 2.5 + 7.0 + torch.randn_like(X) * 2

# create lr model class
class LRModel(nn.Module):
    def __init__(self):
        super(LRModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)
 
# define model hyperparameter
model = LRModel()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()

# training set
epochs = 500
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, Y)
    losses.append(loss)
    loss.backward()
    optimizer.step()
    if ((epoch+1) % 50) == 0:
        print(f"epoch {epoch}, loss {loss:.4f}")

# visualize model prediction
predicted = model(X).detach()
plt.scatter(X, Y, label="Original Data")
plt.plot(X, predicted, color="red", label="Fitted Line")
plt.legend()
plt.title("Linear Regression Model Predictions")
plt.show()