import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import torchvision.transforms as transforms
import cv2  # OpenCV for image processing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel



dataset_path = 'easy_dataset'
image_pixels = 512
neurons = 16
num_samples = 1020
input_size_pca =  image_pixels * image_pixels
learning_rate = 0.1
num_epochs = 150
n_components_pca = 0.90
batch_size = 128
num_classes = 2
input_size = 16

# variables for early stopping
#n_epochs_stop = 3

#Test accuracy: 0.9230769230769231

#dataset_path = 'hard_dataset'

#image_pixels = 512
#neurons = 96
#num_samples = 1020
#input_size_pca = image_pixels * image_pixels
#learning_rate = 0.5
#num_epochs = 150
#n_components_pca = 0.90
#batch_size = 128
#num_classes = 4 
#input_size = neurons

# variables for early stopping
n_epochs_stop = 5

#Test accuracy: 0.8743718592964824


# Define the transform to convert image data to tensors
transform = transforms.Compose([
    transforms.Resize((image_pixels, image_pixels)),  # Resize images for faster computation
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    
    ])

# Load and preprocess the dataset using ImageFolder
dataset_pca = ImageFolder(root=dataset_path, transform=transform)
images_np = np.array([img.numpy().flatten() for img, _ in dataset_pca])
labels = np.array([label for _, label in dataset_pca])


# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=input_size, verbose=1, n_jobs=-1)
clf = clf.fit(images_np, labels)

# Select features based on importance
model = SelectFromModel(clf, prefit=True)
selected_features = model.transform(images_np)

# Convert the selected features back to tensors
selected_features_tensor = torch.from_numpy(selected_features).float()
dataset = [(selected_features_tensor[i], labels[i]) for i in range(len(labels))]

train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)
test_set, val_set = train_test_split(test_set, test_size=0.5, random_state=42)


# Define the dataloaders for training and testing
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=True)
valid_loader = DataLoader(val_set, batch_size, shuffle=True)

# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, neurons)  #TODO change the number of neurons, with a reference
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(neurons, num_classes)  

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Create an instance of the model
model = MLP(input_size, num_classes) 

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)    # TODO change optimizer - SGD, Adam, RMSprop
#weight_decay is the coefficient for the L2 regularization term. It helps prevent overfitting by penalizing large weights.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#DATA FOR PERFORMANCE
train_losses = []
valid_losses = []
train_acc = []



best_valid_loss = float('inf')
epochs_no_improve = 0

# Train the model
for epoch in range(num_epochs):  
    model.train()
    iteration = 0

    for images, labels in train_loader:
        iteration += 1
        optimizer.zero_grad()
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        train_losses.append(train_loss)
        train_loss.backward()
        optimizer.step()
        print(f'Epoch {epoch} Total {iteration / len(train_loader)}, Training Loss: {train_loss:.4f}')



    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)
    print(f'Epoch {epoch}, Validation Loss: {valid_loss:.4f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # If the validation loss hasn't improved for 'patience' epochs, stop training
    if epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        break

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
        print(labels, predicted)

accuracy = correct / total
print('Test accuracy:', accuracy)

#Test accuracy: 0.7973856209150327

#Test accuracy: 0.6928104575163399

#1. Find features to extract, diminish the hypothesis space
#2. As imagens tem fundo preto, e pode confundir o modelo pq partes importantes tbm tem essa cor. Ai teriamos q ou cortar o fundo, ou trocar a cor do fundo
#3. Tem a opcao de fazer um highlight dos tons de cinza, aumentando o contraste
#4. Testar com outros otimizadores, mudar parametros etc (com referencias pra mudanca)