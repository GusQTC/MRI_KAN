import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


# Define the transform to convert image data to tensors

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to
    transforms.ToTensor()  # Then convert them to tensors
])


input_size = 3 * 128 * 128
#image is 512 by 512 rgb
num_classes = 4
# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='dataset', transform=transform)

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Define the dataloaders for training and testing
train_loader = DataLoader(train_set, batch_size=12, shuffle=True)
test_loader = DataLoader(test_set, batch_size=12, shuffle=False)

# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 128)  # Hidden layer with 128 units
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for classification

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Create an instance of the model
model = MLP(input_size, num_classes)  # Replace input_size and num_classes with appropriate values

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):  # Replace 10 with the desired number of epochs
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print('Test accuracy:', accuracy)

#Test accuracy: 0.12863705972434916
