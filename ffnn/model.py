import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons

# create random dataset
x_moon, y_moon = make_moons(
    n_samples=200,
    noise=0.1,
    random_state=42
)
X = torch.tensor(x_moon, dtype=torch.float32)
y = torch.tensor(y_moon, dtype=torch.long)

# create nn class
class ffnn(nn.Module):
    def __init__(self, input_feature, hidden_feature, output_feature):
        super(ffnn, self).__init__()
        self.fc1 = nn.Linear(input_feature, hidden_feature)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_feature, output_feature)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# define model and its hyperparameters
nn_model = ffnn(2, 8, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.01, weight_decay=1e-5)

# training set
epochs = 1000
losses = []
for epoch in range(epochs):
    y_pred = nn_model(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"epoch {epoch+1}/{epochs}, loss: {loss:.4f}")

# plot the model and its dataset
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, x2x2 = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(x2_min, x2_max, 100)
)
grid = torch.tensor(np.c_[xx.ravel(), x2x2.ravel()], dtype=torch.float32)
preds = nn_model(grid).detach().numpy()
preds = np.argmax(preds, axis=1)

plt.contour(xx, x2x2, preds.reshape(xx.shape), cmap="RdBu", alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=20)
plt.title("Decision Boundary")
plt.show()