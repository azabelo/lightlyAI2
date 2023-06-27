import copy
import torchvision
from torch import nn
import torchvision.transforms as transforms
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, activate_requires_grad
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import wandb
from torch.optim import Adam
from torch.optim import SGD


class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

def pretrain():

   # torch.multiprocessing.freeze_support()

    # Define the data transformation (not sure if normalization helps)
    # transform = transforms.Compose(
    #     [transforms.Resize((224,224)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    transform = DINOTransform()
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    model = DINO(backbone, input_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    crop_transform = DINOTransform(global_crop_size=196, local_crop_size=64)
    dataset = LightlyDataset.from_torch_dataset(cifar10, transform=crop_transform)

    collate_fn = MultiViewCollate()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = DINOLoss(
        output_dim=2048,
        warmup_teacher_temp_epochs=5,
    )
    # move loss to correct device because it also contains parameters
    criterion = criterion.to(device)

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=1e-6)

    # define the lr scheduler
    #scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2, verbose=False)

    # Initialize W&B
    wandb.init(project='unsupervised-pretraining')
    # Track the model and the hyperparameters
    wandb.watch(model)

    #DINO pretraining
    epochs = 15
    print("Starting Training")
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        print("epoch: ", epoch)
        count = 0
        for views, _, _ in dataloader:
            count += 1
            if count%100 == 0:
                print(count)
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)
            views = [view.to(device) for view in views]
            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)

            # Log the loss after every batch
            wandb.log({"train_loss": loss})

            total_loss += loss.detach()
            loss.backward()
            # We only cancel gradients of student head.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        #scheduler.step(avg_loss)

        # checkpointing our model
        file = "unsupervised_pretraining/loss" + str(avg_loss)
        torch.save(model.teacher_backbone.state_dict(), file)
        print("saved model state at epoch ", epoch)

    # Finish the run
    wandb.finish()
   #we need to reactivate requires grad to perform supervised backpropagation later
    activate_requires_grad(model.teacher_backbone)
    return model.teacher_backbone


def create_datasets(config=None):
    # apply transformations to the train and test
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

def train(model=None, config=None):

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        train_loader, val_loader = create_datasets(config)

        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        learning_rate = config.learning_rate
        # Define the optimizer
        if config.optimizer == 'Adam':
            optimizer = Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = SGD(model.parameters(), lr=learning_rate)

        #define the lr scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor = 0.1, patience=2, verbose=True)

        # Move the model to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Track the model and the hyperparameters
        wandb.watch(model)

        # Training loop
        num_epochs = 20
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


def train_once(model):
    print("starting supervised training")

    config = {
        'batch_size': 32,
        'learning_rate': 1e-5,
        'optimizer': 'Adam',
        'augmentations': False
    }

    # Initialize wandb
    wandb.init(project='pretrained', config=config)
    train(model, config)

    # Finish the run
    wandb.finish()


pretrained_model = pretrain()

# model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
# pretrained_model = model.load_state_dict(torch.load("unsupervised_pretraining/losstensor(3.9870, device='cuda:0')"))
train_once(pretrained_model)
