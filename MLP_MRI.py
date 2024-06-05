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
    transforms.Resize((512, 512)),  # Resize all images to
    transforms.ToTensor(),  # Then convert them to tensors
    transforms.Grayscale(num_output_channels=1),  # Convert the images to grayscale
])


input_size =  512 * 512  # Image is 3 channels (RGB) and 512x512 image size
learning_rate = 0.05
num_epochs = 15
batch_size = 64
#image is 512 by 512 rgb
num_classes = 2
# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='new_dataset', transform=transform)# 1020 images

# Split the dataset into training and testing sets
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
        self.fc1 = nn.Linear(input_size, 64)  # Reduced the number of units in the hidden layer
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(64, num_classes)  

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
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


# variables for early stopping
n_epochs_stop = 5
best_valid_loss = float('inf')
epochs_no_improve = 0

# Train the model # TODO change to train function
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

accuracy = correct / total
print('Test accuracy:', accuracy)

#Test accuracy: 0.7843137254901961


#1. Find features to extract, diminish the hypothesis space
#2. As imagens tem fundo preto, e pode confundir o modelo pq partes importantes tbm tem essa cor. Ai teriamos q ou cortar o fundo, ou trocar a cor do fundo
#3. Tem a opcao de fazer um highlight dos tons de cinza, aumentando o contraste
#4. Testar com outros otimizadores, mudar parametros etc (com referencias pra mudanca)