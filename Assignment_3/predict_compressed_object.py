import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import json
import urllib.request
import os
import matplotlib.pyplot as plt

# Download the ImageNet class labels if not present
def load_imagenet_classes():
    if not os.path.exists("imagenet_class_index.json"):
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        urllib.request.urlretrieve(url, "imagenet_class_index.json")
    
    with open("imagenet_class_index.json", "r") as f:
        class_idx = json.load(f)
    return {int(key): value[1] for key, value in class_idx.items()}

# Add Gaussian noise to the image
def add_gaussian_noise(image, mean=0.0, std=1.0):
    # Convert image to numpy array
    image_np = np.array(image) / 255.0  # Normalize to [0,1]
    noise = np.random.normal(mean, std, image_np.shape)  # Generate Gaussian noise
    noisy_image_np = image_np + noise  # Add noise
    noisy_image_np = np.clip(noisy_image_np, 0.0, 1.0)  # Clip values to [0,1]
    noisy_image = Image.fromarray((noisy_image_np * 255).astype(np.uint8))  # Convert back to image
    return noisy_image

# Autoencoder 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

# Load pretrained ResNet model
def load_resnet_model():
    resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  
    resnet_model.eval()  # Set to evaluation mode
    return resnet_model

# Main function to perform the task
def main():
    # Load image and apply Gaussian noise
    img_path = 'Assignment3_dog_to_compress.jpg'
    image = Image.open(img_path).convert('RGB')
    plt.imshow(image)
    plt.title("Original Image")
    plt.show()
    
    noisy_image = add_gaussian_noise(image)
    plt.imshow(noisy_image)
    plt.title("Noisy Image")
    plt.show()

    # Transform the image to tensor and normalize for Autoencoder and ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
    ])
    noisy_image_tensor = transform(noisy_image).unsqueeze(0)  # Add batch dimension

    # Compress the noisy image using Autoencoder
    autoencoder = Autoencoder()
    autoencoder.eval()  # Set to evaluation mode
    with torch.no_grad():
        compressed_image = autoencoder(noisy_image_tensor)
    
    # Reconstruct the image from the compressed representation
    reconstructed_image = compressed_image.squeeze(0).permute(1, 2, 0).numpy()
    reconstructed_image = np.clip(reconstructed_image, 0, 1)  # Ensure valid image range
   
    # Convert reconstructed image back to PIL for transformation
    reconstructed_image_pil = Image.fromarray((reconstructed_image * 255).astype(np.uint8))
    
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed Image")
    plt.show()

    # Classify the reconstructed image using the pretrained ResNet model
    resnet_model = load_resnet_model()
    with torch.no_grad():
        reconstructed_image_tensor = transform(reconstructed_image_pil).unsqueeze(0)  # Add batch dimension
        output = resnet_model(reconstructed_image_tensor)
    
    # Get predicted class index
    _, predicted_class = torch.max(output, 1)
    
    # Load ImageNet classes
    class_idx = load_imagenet_classes()
    
    # Map the predicted index to class name
    predicted_class_name = class_idx[predicted_class.item()]
    print(f"Predicted class: {predicted_class_name}")

if __name__ == "__main__":
    main()
