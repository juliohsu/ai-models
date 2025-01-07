import torch
import torch.nn as nn
import torch.optim as optimizer
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(30),
    #transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

valid_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=train_transforms, download=False)
valid_dataset = datasets.MNIST(root="./data", train=False, transform=valid_transforms, download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        #self.conv5 = nn.Conv2d(128, 256, kernel_size=6, stride=2, padding=4)
        #self.bn5 = nn.BatchNorm2d(256)
        #self.conv6 = nn.Conv2d(256, 512, kernel_size=6, stride=2, padding=2)
        #self.bn6 = nn.BatchNorm2d(512)
        #self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        #self.pool3 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        #x = self.relu(self.bn5(self.conv5(x)))
        #x = self.relu(self.bn6(self.conv6(x)))
        #x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        return x

model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optim = optimizer.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optim, step_size=2, gamma=0.7)
epochs = 5

for epoch in range(epochs):
    model.train()
    running_losses = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        losses = criterion(preds, labels)
        optim.zero_grad()
        losses.backward()
        optim.step()
        running_losses += losses.item()
    print(f"Epoch {epoch+1}, losses {running_losses}")
    scheduler.step()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        _, predictions = torch.max(preds, 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    print(f"total of {total}, getting {correct} correct answers")

examples = iter(valid_loader)
images, labels = next(examples)
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    preds = model(images)
    _, predictions = torch.max(preds, 1)
num_set = 1
num_img = 6
fig, axes = plt.subplots(num_set, num_img, figsize=(12, 4))
for j in range(num_img):
    axes[j].imshow(images[j].cpu().squeeze(), cmap="gray")
    axes[j].set_title(f"Prediction: {predictions[j].item()}")
    axes[j].axis("off")
plt.show()