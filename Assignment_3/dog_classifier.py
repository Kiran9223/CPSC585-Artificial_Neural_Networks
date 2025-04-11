# dog_classifier.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image  # Added for image processing

# Define data transformations including resizing and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard size for many CNNs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Load dataset from the 'Dogs' directory
dataset = datasets.ImageFolder(root='Dogs', transform=transform)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

# Split dataset into training and testing sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define a simple residual block for skip connections
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return x + self.conv(x)

# Define the CNN model
class DogClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogClassifier, self).__init__()
        # First convolutional block: 3 -> 32 channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 112x112
        )
        # Second block: 32 -> 64 channels
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 56x56
        )
        # Residual block to enhance gradient flow and feature reuse
        self.resblock = ResidualBlock(64)
        # Third block: 64 -> 128 channels
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 28x28
        )
        # Fully connected layer for classification
        self.fc = nn.Linear(128 * 28 * 28, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.resblock(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # Flatten feature maps
        out = self.fc(out)
        return out

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)
    model = DogClassifier(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10  # Number of training epochs
    start_time = time.time()  # Record training start time
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / train_size
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    total_training_time = time.time() - start_time
    
    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    
    print(f"\nTotal training time: {total_training_time:.2f} seconds")
    print(f"Classification accuracy on test set: {accuracy:.2f}%")
    
    # --- New Section: Load and Test a Custom Image ---
    image_path = "dog.jpg"
    if os.path.exists(image_path):
        # Load the image and apply the same transform as used for testing
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
        
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
        
        # Map the prediction to the corresponding class name
        predicted_class = dataset.classes[predicted.item()]
        print(f"\nPredicted class for '{image_path}': {predicted_class}")
    else:
        print(f"\nImage file '{image_path}' not found.")

if __name__ == '__main__':
    main()
