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
import matplotlib.pyplot as plt
import numpy as np

num_samples = 1020
input_size_pca =  128 * 128  # Image is 3 channels (RGB) and 512x512 image size #TODO change input for reducing size
learning_rate = 0.05
num_epochs = 25
n_components_pca = 0.8
batch_size = 96 # increasing doesnt improve accuracy,  it increases the time to traib
#image is 512 by 512 rgb
num_classes = 2 #TODO change to 4 classes
input_size = n_components_pca



def apply_pca(image_tensor):
    # Convert the PyTorch tensor to a NumPy array
    image_np = image_tensor.numpy()
    
    # Reshape the image to a 2D array (flattening)
    original_shape = image_np.shape
    image_np = image_np.reshape(original_shape[0], -1)
    
    # Apply PCA with n_components set to 0.8
    pca = PCA(n_components=0.8)
    pca.fit(image_np)
    transformed_image_np = pca.transform(image_np)
    
    # Convert the NumPy array back to a PyTorch tensor
    # Note: The shape of the tensor will be different from the original
    transformed_image_tensor = torch.from_numpy(transformed_image_np)
    
    print(f'Original shape: {original_shape}, Transformed shape: {transformed_image_tensor.shape}')
    
    return transformed_image_tensor

# Define the transform to convert image data to tensors
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images for faster computation
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    
    ])

# Load and preprocess the dataset using ImageFolder
dataset = ImageFolder(root='new_dataset', transform=transform)

images = np.array([img.flatten() for img, _ in dataset])

#pca = PCA()
#pca.fit(images)

#show images
for image in images:
    plt.imshow(image.reshape(128, 128), cmap='gray')
    plt.show()

'''
# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid(True)
plt.show()

# Find the optimal number of components for 95% variance
optimal_components = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
print(f'Optimal number of components: {optimal_components}')
'''