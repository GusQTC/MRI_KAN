import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from kan import KAN
import numpy as np


torch.set_default_dtype(torch.float32)

# Define the transform to convert image data to tensors


#
class ReshapeTransform:
    def __call__(self, img):
        return img.view(-1)


# Add the ReshapeTransform to your existing transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 128x128 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #ReshapeTransform()  # Flatten each image tensor for 2 equal to subscipt in einsum
])

# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='dataset', transform=transform)


input_size = 3 * 256 * 256
#image is 512 by 512 rgb
num_classes = 4
batch_size = 72
num_epochs = 2

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Define the dataloaders for training and testing
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)
# In KAN its as below
#  dataset : dic
               # contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_...
# THerefore, we need to change the dataset for this new format



images_train, labels_train = next(iter(train_loader))
images_test, labels_test = next(iter(test_loader))
# Flatten the images and labels
train_inputs = images_train.view(images_train.size(0), -1)
train_labels = labels_train.view(labels_train.size(0), -1)
test_inputs = images_test.view(images_test.size(0), -1)
test_labels = labels_test.view(labels_test.size(0), -1)

# Create the dictionary for KAN training
dataset_kan = {
    'train_input': train_inputs,
    'train_label': train_labels,
    'test_input': test_inputs,
    'test_label': test_labels
}


# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 225)  # Corrected input size here
        self.kan = KAN(width=[225, 225, 4], grid=5, k=3, seed=0)


    def forward(self, x):
        x = self.pool(F.relu(self.conv(x))) # Convolution + ReLU + Pooling -> feature maps of size (batch_size, 32, 256, 256).
        #print(f"Convolutional layer output: {x.shape}")

        x = x.view(x.size(0), -1) # Flatten into 2D tensor of size (batch_size, 32256256)
        #$print(f"Flatenning layer output: {x.shape}")

        x = F.relu(self.fc1(x)) # Fully connected layer + ReLU

        x = torch.squeeze(x) # Squeeze to reduce unnecessary dimensions
        #print(f"Squeeze layer output: {x.shape}")

        return self.kan(x) # KAN

model = MyModel()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def analyse_loader(loader):
    for i, (images, labels) in enumerate(loader):
        # Print the first image and its corresponding label
        print(f"Sample {i}: Image shape: {images.shape}, Label: {labels}")
        
        # Optionally, if you want to visualize the image, you can use matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(images[0].permute(1, 2, 0))  # Adjust the permute order based on your image channel layout
        plt.title(f"Label: {labels[0]}")
        plt.show()
        
        # Break after printing the first sample
        break
#analyse_loader(train_loader)
# Train the model
#train(model, optimizer, criterion, train_loader, steps=10)
model.kan.train(dataset_kan, opt='LBFGS', steps=50, lamb=0.01)
#test_model(model, criterion, test_loader)
