from PIL import Image

# Open the image using PIL
image = Image.open('dataset\glioma_tumor\gg (1).jpg')  # Replace with the path to your image

# Get the number of channels, height, and width of the image
channels = image.mode
width, height = image.size

print('Number of channels:', channels)
print('Image width:', width)
print('Image height:', height)