# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import copy

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms

from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import torch
import wandb
from torch.optim import Adam

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # Define the data transformation
    transform = transforms.Compose(
        [transforms.ToTensor(),  # Converts PIL image or numpy.ndarray to torch.Tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize the image tensor


    class DINO(torch.nn.Module):
        def __init__(self, backbone, input_dim):
            super().__init__()
            self.student_backbone = backbone
            self.student_head = DINOProjectionHead(
                input_dim, 128, 64, 32, freeze_last_layer=1
            )
            self.teacher_backbone = copy.deepcopy(backbone)
            self.teacher_head = DINOProjectionHead(input_dim, 128, 64, 32)
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


    # from nanoViT.src import model
    # from nanoViT.src.model import VisionTransformer
    #
    # # pretrained model params
    # custom_config = {
    #     "img_size": 32,
    #     "in_chans": 3,
    #     "patch_size": 8,
    #     "embed_dim": 64,
    #     "depth": 1,
    #     "n_heads": 1,
    #     "qkv_bias": True,
    #     "mlp_ratio": 4,
    # }
    #
    # backbone = VisionTransformer(**custom_config)
    # print(type(backbone))


    #
    # resnet = torchvision.models.resnet18()
    # backbone = nn.Sequential(*list(resnet.children())[:-1])
    input_dim = 32
    # instead of a resnet you can also use a vision transformer backbone as in the
    # original paper (you might have to reduce the batch size in this case):

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
    input_dim = backbone.embed_dim
    print(type(backbone))


    model = DINO(backbone, input_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    print(type(cifar10))

    # # we ignore object detection annotations by setting target_transform to return 0
    # pascal_voc = torchvision.datasets.VOCDetection(
    #     "datasets/pascal_voc", download=True, target_transform=lambda t: 0
    # )



    transform = DINOTransform(global_crop_size=32)
    dataset = LightlyDataset.from_torch_dataset(cifar10, transform=transform)

    print(type(dataset))
    # or create a dataset from a folder containing images or videos:
    # dataset = LightlyDataset("path/to/folder")

    collate_fn = MultiViewCollate()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    criterion = DINOLoss(
        output_dim=32,
        warmup_teacher_temp_epochs=5,
    )
    # move loss to correct device because it also contains parameters
    criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 0

    print("Starting Training")
    for epoch in range(epochs):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
        print("here")
        print(len(dataloader))
        count = 0
        for views, _, _ in dataloader:
            count += 1
            print(count)
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum_val)
            update_momentum(model.student_head, model.teacher_head, m=momentum_val)
            views = [view.to(device) for view in views]
            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.detach()
            loss.backward()
            # We only cancel gradients of student head.
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")



    #now that we've finished pretraining, lets train it to classify the images

    #also make a new wandb project
    # Initialize W&B
    wandb.init(project='unsupervised-pretrained-VIT')

    pretrained_model = model.teacher_backbone

    # Create the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(cifar10))
    val_size = len(cifar10) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(cifar10, [train_size, val_size])

    # Define the data loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=1e-5)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track the model and the hyperparameters
    wandb.watch(model)
    wandb.config.update({"learning_rate": 1e-5, "batch_size": batch_size})

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        # Log validation metrics after each epoch
        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

        # Print training and validation metrics for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}]\t"
              f"Val Loss: {val_loss:.4f}\t"
              f"Val Accuracy: {val_accuracy:.4f}")

    # Finish the run
    wandb.finish()

