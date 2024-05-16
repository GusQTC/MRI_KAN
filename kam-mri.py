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

class ReshapeTransform:
    def __call__(self, img):
        return img.view(-1)


# Add the ReshapeTransform to your existing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    #ReshapeTransform()  # Flatten each image tensor for 2 equal to subscipt in einsum
])

# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='dataset', transform=transform)


input_size = 3 * 128 * 128
#image is 512 by 512 rgb
num_classes = 4
batch_size = 72
num_epochs = 10
# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='dataset', transform=transform)

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

# Define the dataloaders for training and testing
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)
images, labels = next(iter(train_loader))

# Define the neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 25)  # Corrected input size here
        self.kan = KAN(width=[25, 25, 4], grid=5, k=3, seed=0)


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
def test_model(model, criterion, dataloader):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():  # No need to track gradients during testing
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            # Update running loss
            running_loss += loss.item()
            
            # Count correct predictions
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Training loop
def train(model, optimizer, criterion, train_loader, steps):
    for epoch in range(num_epochs): # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()                
                # Optionally, print statistics
                running_loss += loss.item()
                print(f'Epoch: {epoch+1}, Loss: {running_loss}, Percent Complete: {i/len(train_loader)}')
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

# Train the model
train(model, optimizer, criterion, train_loader, steps=10)
test_model(model, criterion, test_loader)

# Perform automatic symbolic regression
#lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
#model.auto_symbolic(lib=lib)

# Get the symbolic formulas
#formula1, formula2 = model.symbolic_formula()[0]

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
#print('Train accuracy of the formula:', acc(formula1, formula2, train_loader))
#print('Test accuracy of the formula:', acc(formula1, formula2, test_loader))