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
    transforms.Resize((256, 256)),  # Resize all images to
    transforms.ToTensor(),  # Then convert them to tensors
    transforms.Grayscale(num_output_channels=1),  # Convert the images to grayscale
])


input_size = 256 * 256  # 3 channels (RGB) and 512x512 image size
learning_rate = 0.05
num_epochs = 15
batch_size = 96
#image is 512 by 512 rgb
num_classes = 2
# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='new_dataset', transform=transform)# 1020 images

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Define the dataloaders for training and testing
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 256)  # Hidden layer with 256 units
        self.relu = nn.Sigmoid()                  # TODO test with sigmoid
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)  # Output layer for classification

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
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # TODO change optimizer - SGD, Adam, RMSprop
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the model # TODO change to train function
for epoch in range(num_epochs):  # Replace 10 with the desired number of epochs
    model.train()
    iteration = 0
    for images, labels in train_loader:
        iteration += 1
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch} Percent {iteration / len(train_loader)}, Loss: {loss:.4f}')

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

#Test accuracy: 0.6519607843137255


#1. Resize the dataset images ( 3 * 512 * 512) - pca, grayscale
#2. As imagens tem fundo preto, e pode confundir o modelo pq partes importantes tbm tem essa cor. Ai teriamos q ou cortar o fundo, ou trocar a cor do fundo
#3. Tem a opcao de fazer um highlight dos tons de cinza, aumentando o contraste
###4. Separar lado da imagem
#5. Testar com outros otimizadores