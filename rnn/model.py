import torch
import torch.nn as nn
import torch.optim as optimizer

from torch.utils.data import DataLoader

# configure the gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a custom dataset
data = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]
]

labels = [0, 1, 0]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index].to(device), self.labels[index].to(device)

# configure custom dataset
data, labels = torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# create a custom rnn model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, h0=None):
        if h0 == None:
            h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        x, _ = self.rnn(x, h0)
        x = self.fc(x[:, -1, :])
        return x

# configure custom rnn model
model = RNN(12, 8, 2)
optim = optimizer.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# train
for epoch in range(5):
    epoch_losses = 0.0
    for loader_data, loader_labels in dataloader:
        loader_data = torch.nn.functional.one_hot(loader_data, num_classes=12).float()
        preds = model(loader_data)
        losses = criterion(preds, loader_labels)
        epoch_losses += losses
        optim.zero_grad()
        losses.backward()
        optim.step()
    print(f"epoch {epoch}, total losses: {epoch_losses:.4f}")

# valid
with torch.no_grad():
    for zipper_data, zipper_label in zip(data, labels):
        zipper_data = torch.nn.functional.one_hot(zipper_data, num_classes=12).float()
        zipper_data = zipper_data.unsqueeze(0)
        zipper_data = zipper_data.to(device)
        preds = model(zipper_data)
        prediction = torch.argmax(preds, 1).item()
        print(f"For the data, {zipper_data}, we got prediction {prediction} and its answer {zipper_label}")