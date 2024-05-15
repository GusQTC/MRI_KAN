import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from kan import KAN
import numpy as np


torch.set_default_dtype(torch.float32)

# Define the transform to convert image data to tensors

class ReshapeTransform:
    def __call__(self, img):
        return img.view(-1)


# Add the ReshapeTransform to your existing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    ReshapeTransform()  # Flatten each image tensor for 2 equal to subscipt in einsum
])

# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='dataset', transform=transform)


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
images, labels = next(iter(train_loader))
print(images.shape)

# Define the neural network architecture
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)


# Define the optimizer and loss function
optimizer = optim.LBFGS(model.parameters())
criterion = nn.CrossEntropyLoss()

def train_acc(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def test_acc(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Training loop
def train(model, optimizer, criterion, train_loader, steps):
    for step in range(steps):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Step: {step+1}, Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train(model, optimizer, criterion, train_loader, steps=20)

# Perform automatic symbolic regression
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
model.auto_symbolic(lib=lib)

# Get the symbolic formulas
formula1, formula2 = model.symbolic_formula()[0]

# Define a function to compute the accuracy of the formulas
def acc(formula1, formula2, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        batch = images.shape[0]
        for i in range(batch):
            logit1 = np.array(formula1.subs({'x_1': images[i, 0], 'x_2': images[i, 1]})).astype(np.float64)
            logit2 = np.array(formula2.subs({'x_1': images[i, 0], 'x_2': images[i, 1]})).astype(np.float64)
            correct += (logit2 > logit1) == labels[i]
            total += 1
    return correct / total

# Compute the accuracy of the formulas on the train and test sets
print('Train accuracy of the formula:', acc(formula1, formula2, train_loader))
print('Test accuracy of the formula:', acc(formula1, formula2, test_loader))