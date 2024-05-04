import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
from customDataset import CropsDataset
import numpy as np
from matplotlib import pyplot as plt
import random

label_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/quantitative analysis/binary classifier/crop.csv'
image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/images'

in_channel = 3
n_classes = 2
batch_size = 1
epochs = 4

# Preparing the data for training ResNet18 model:
# ----------------------------------------------

transform = transforms.Compose([transforms.ToTensor(),  # Convert image to tensor.
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = CropsDataset(label_file=label_path, image_directory=image_path,
                       transform=transform)  # A custom torch Dataset instance

indices = [random.choice(range(len(dataset))) for i in range(20)]  # Getting indices for sampling for the Subset method
test_set = torch.utils.data.Subset(dataset, indices)  # Sampling n images from the dataset fot testing
train_set, val_set = torch.utils.data.random_split(dataset, [0.80, 0.20])  # Splitting the data in train and validation

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(train_set)
valid_data_size = len(val_set)
test_data_size = len(test_set)
print(f'Training data size: {train_data_size}, Validation data size: {valid_data_size}, Test data size: {test_data_size}')

image_datasets = {'train': train_set, 'val': val_set}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
               for x in ['train', 'val']}  # Getting Dataloader instance from dataset instance
test_loader = DataLoader(dataset=test_set)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Loading the FCResNet18 model:
# ----------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def save_models(epochs, model):
    torch.save(model.state_dict(), "custom_model{}.model".format(epochs))
    print("Checkpoint Saved")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    # since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train' and epoch_acc > best_acc:
                save_models(epoch, model)
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


class FCNLayer(torch.nn.Module):
    def __init__(self):
        super(FCNLayer, self).__init__()
        self.classifier = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1)

    def forward(self, x):
        return self.classifier(x)


fcnlayer = FCNLayer()
model_ft = models.resnet18(pretrained=True)
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.fc = fcnlayer

# ----------------------------------------------------------------------------------------------------------------------
# The initial way dataloader was prepared:
label_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/quantitative analysis/binary classifier/BC-ResNet18/crop.csv'
image_path = '/Users/frankygeorge/PhD/PhD-Works/Pedestrian Project/PedestrianProject/data/crops/images'

in_channel = 3
n_classes = 2
batch_size = 1
epochs = 5

# Preparing the data for training ResNet18 model:
# ----------------------------------------------

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = CropsDataset(label_file=label_path, image_directory=image_path,
                       transform=transform)  # A custom torch Dataset instance

indices = [random.choice(range(len(dataset))) for i in range(20)]  # Getting indices for sampling for the Subset method
test_set = torch.utils.data.Subset(dataset, indices)  # Sampling n images from the dataset fot testing
train_set, val_set = torch.utils.data.random_split(dataset, [0.80, 0.20])  # Splitting the data in train and validation

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(train_set)
valid_data_size = len(val_set)
test_data_size = len(test_set)
print(f'Training data size: {train_data_size}, Validation data size: {valid_data_size}, Test data size: {test_data_size}')

train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                          shuffle=True)  # Getting Dataloader instance from dataset instance
test_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_set)

for i in range(2):
    for input, label in train_loader:
        print(input)
        print(label)
        print()

# Loading the FCResNet18 model:
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU: " + torch.cuda.get_device_name(device))
else:
    print("Using CPU")

# model = FCResNet18(pretrained=True)
# model.to(device)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# # Freeze model parameters
# for param in model.parameters():
#     param.requires_grad = False
