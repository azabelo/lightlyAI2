import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Adam
import wandb

# Initialize W&B
wandb.init(project='supervised vision transformer')

# Load the Vision Transformer model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)

# Create the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])

cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(cifar10))
val_size = len(cifar10) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(cifar10, [train_size, val_size])

# Define the data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = Adam(model.parameters(), lr=1e-4)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Track the model and the hyperparameters
wandb.watch(model)
wandb.config.update({"learning_rate": 1e-4, "batch_size": batch_size})

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    # Compute the average training loss for the epoch
    train_loss /= len(train_dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute the average validation loss and accuracy for the epoch
        val_loss /= len(val_dataset)
        val_accuracy = correct / total

    # Log metrics to Weights & Biases
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})

    # Print training and validation metrics for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}]\t"
          f"Train Loss: {train_loss:.4f}\t"
          f"Val Loss: {val_loss:.4f}\t"
          f"Val Accuracy: {val_accuracy:.4f}")

# Finish the run
wandb.finish()