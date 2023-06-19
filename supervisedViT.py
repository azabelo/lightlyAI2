import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.optim import Adam
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy
import matplotlib.pyplot as plt
import hashlib
import random

def create_datasets(config=None):
    # apply transformations to the train and test
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    if config.augmentations:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=192),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
    ])

    # Create the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    print("train size: ", len(train_dataset), "val size: ", len(val_dataset))

    # image, _ = train_dataset[0]
    # image_np = image.permute(1,2,0).numpy()
    # plt.imshow(image_np)
    # plt.show()

    # Define the data loaders
    batch_size = config.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = create_datasets(config)

        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Load the Vision Transformer model
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)

        learning_rate = config.learning_rate
        # Define the optimizer
        if config.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = SGD(model.parameters(), lr=learning_rate)

        #define the lr scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor = 0.1, patience=3, verbose=True)

        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Track the model and the hyperparameters
        wandb.watch(model)
       # wandb.config.update({"learning_rate": 1e-5, "batch_size": batch_size})

        # Training loop
        num_epochs = 1
        print("batch count: ", len(train_loader))
        for epoch in range(num_epochs):
            model.train()
            print("epoch: ", epoch)

            for batch_idx, (images, labels) in enumerate(train_loader):
                if (batch_idx+1)%100 == 0:
                    print("batch ",batch_idx+1)

                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log the loss after every batch
                wandb.log({"train_loss": loss.item()})

            # Validation loop
            model.eval()
            val_loss = 0
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
                val_loss /= len(val_loader)
                val_accuracy = correct / total

            # Log validation metrics after each epoch
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
            #log weight histogram
            # wandb.log({"Gradients": wandb.Histogram(model.parameters())})

            # Print training and validation metrics for the epoch
            print(f"Epoch [{epoch+1}/{num_epochs}]\t"
                  f"Val Loss: {val_loss:.4f}\t"
                  f"Val Accuracy: {val_accuracy:.4f}")

            scheduler.step(val_loss)
            # checkpointing our model
            file = "supervisedVIT_untrained_epoch" + str(epoch)
            torch.save(model.state_dict(), file)
            print("saved model state at epoch ", epoch)

# Initialize W&B
#wandb.init(project='supervised VIT (sweep)')

# Define the hyperparameters for grid search
batch_sizes = [16, 32, 64]
optimizers = ['Adam', 'SGD']
learning_rates = [1e-4, 1e-5, 1e-6]
augmentations = [True, False]

# Define the sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {
            'values': batch_sizes
        },
        'optimizer': {
            'values': optimizers
        },
        'learning_rate': {
            'values': learning_rates
        }
        ,
        'augmentations': {
            'values': augmentations
        }
    }
}

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train)


# Finish the run
wandb.finish()