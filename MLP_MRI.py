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


image_pixels = 128
neurons = 8
num_samples = 1020
input_size_pca =  image_pixels * image_pixels  # Image is 3 channels (RGB) and 512x512 image size #TODO change input for reducing size
learning_rate = 0.05
num_epochs = 25
n_components_pca = 0.96
batch_size = 96 # increasing doesnt improve accuracy,  it increases the time to train
#image is 512 by 512 rgb
num_classes = 2 #TODO change to 4 classes
input_size = 96


def contrast_normalization(image_np):
    "Normalization for the contrast of the image, to improve the quality of the image for the model, as the black background can affect the performance"
    # Assuming image_np is a 2D numpy array for grayscale images
    # Use OpenCV's equalizeHist function for histogram equalization
    equalized_image = np.zeros_like(image_np)
    for i in range(image_np.shape[0]):
        equalized_image[i] = cv2.equalizeHist(image_np[i].astype(np.uint8))
    return equalized_image

def apply_pca_int(image_tensors, n_components=input_size):
    "Trying to apply the pca with fixed number of components, basically already defining the number of features. For that to work, we need to apply pca to all images together"
    # Convert the PyTorch tensor to a NumPy array
    images_np = image_tensors.numpy()
    
    # Normalize the image to have pixel values between 0 and 1
    #image_np = image_np / 255.0
    
    
    # Apply contrast normalization
    #images_np = contrast_normalization(images_np)
    
    # Flatten each image in the batch
    channels, height, width = images_np.shape
    images_np_flattened = images_np.reshape(channels, -1)
    #(1, 512, 512)
    
    # Apply PCA
    pca = PCA(n_components=n_components, svd_solver='auto')
    pca_result = pca.fit_transform(images_np_flattened)
    print(pca_result.shape)
    
    # Convert the result back to a PyTorch tensor
    transformed_images_tensor = torch.from_numpy(pca_result)
    return transformed_images_tensor

def apply_pca_float(image_tensor, n_components=n_components_pca, input_nn=input_size):
    "Applying pca to preserve a certain amount of variance, and then truncating or padding the result to have the desired number of components"
    # Convert the PyTorch tensor to a NumPy array
    image_np = image_tensor.numpy()
    
    # Normalize the image to have pixel values between 0 and 1
    image_np = image_np / 255.0
    
    # Apply contrast normalization
    image_np = contrast_normalization(image_np)
    
    # Reshape the image to a 2D array (flattening)
    original_shape = image_np.shape
    images_np_flattened = image_np.reshape(original_shape[0], -1)
    
        # Normalize your data
    scaler = StandardScaler()
    images_np_flattened = scaler.fit_transform(images_np_flattened)  # Normalize the data



    # Apply PCA with n_components set to the optimal number
    pca = PCA(n_components=n_components, svd_solver='auto')
    pca_result = pca.fit_transform(images_np_flattened)
    
        # If the PCA result does not have the desired shape, pad or truncate
    if pca_result.shape[1] < input_nn:
        # Pad the PCA result to have the desired number of components
        pca_result_padded = np.pad(pca_result, ((0, 0), (0, input_nn - pca_result.shape[1])), 'constant', constant_values=0)
        transformed_image_np = pca_result_padded
    elif pca_result.shape[1] > input_nn:
        # Truncate the PCA result to have the desired number of components
        transformed_image_np = pca_result[:, :input_nn]
    else:
        transformed_image_np = pca_result
    
    # Convert the NumPy array back to a PyTorch tensor
    transformed_image_tensor = torch.from_numpy(transformed_image_np)
    return transformed_image_tensor

# Define the transform to convert image data to tensors
transform = transforms.Compose([
    transforms.Resize((image_pixels, image_pixels)),  # Resize images for faster computation
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(apply_pca_float)
    
    ])

# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='new_dataset', transform=transform)


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
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(neurons, num_classes)  

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)    # TODO change optimizer - SGD, Adam, RMSprop
#weight_decay is the coefficient for the L2 regularization term. It helps prevent overfitting by penalizing large weights.
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#DATA FOR PERFORMANCE
train_losses = []
valid_losses = []
train_acc = []


# variables for early stopping
n_epochs_stop = 3
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

accuracy = correct / total
print('Test accuracy:', accuracy)

#Test accuracy: 0.7973856209150327

#Test accuracy: 0.6928104575163399

#1. Find features to extract, diminish the hypothesis space
#2. As imagens tem fundo preto, e pode confundir o modelo pq partes importantes tbm tem essa cor. Ai teriamos q ou cortar o fundo, ou trocar a cor do fundo
#3. Tem a opcao de fazer um highlight dos tons de cinza, aumentando o contraste
#4. Testar com outros otimizadores, mudar parametros etc (com referencias pra mudanca)