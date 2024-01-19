''' 1. Train
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg16"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
''' 

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
# import functions_train
# import functions_predict

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument('data_dir', nargs = '*', action = 'store', default = './flowers/')
parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'gpu')
parser.add_argument('--save_dir', dest = 'save_dir', action='store', default = './checkpoint.pth')
parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.01)
# parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument('--epochs', dest = 'epochs', action = 'store', type = int, default = 5)
parser.add_argument('--arch', dest = 'arch', action = 'store', default = 'vgg16', type = str)
parser.add_argument('--hidden_units', type = int, dest = 'hidden_units', action = 'store', default = 120)

#setting values data loading
args = parser.parse_args()
where = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
# dropout = args.dropout

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if torch.cuda.is_available() and power == 'gpu':
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
def nn_measure(structure='vgg16', dropout=0.5, hidden_layer1=120, lr=0.001):
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216  # Update the input size for alexnet
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024  # Update the input size for densenet121
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088  # Update the input size for vgg16
    else:
        print("Im sorry but {} is not a valid model.".format(structure))

    for param in model.parameters():
        param.requires_grad = False

    # Adjust the input size in the first linear layer to match the output size of the features
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layer1)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer1, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device)

    return model, optimizer, criterion


def loadData(data_dir = "./flowers"):
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    data_dir = str(data_dir).strip('[]').strip("'")
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    cropped_size = 224
    resized_size = 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # random scaling, cropping, and flipping, resized to 224x224 pixels
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(cropped_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])
    
    # resize then crop the images to the appropriate size
    validate_transforms = transforms.Compose([transforms.RandomResizedCrop(cropped_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(means, stds)])
    
    # resize then crop the images to the appropriate size
    test_transforms = transforms.Compose([transforms.Resize(resized_size), # Why 255 pixels? 
                                          transforms.CenterCrop(cropped_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform = validate_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    image_data = [train_data, validate_data, test_data] 
    batch_size = 64
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
    validate_loader = torch.utils.data.DataLoader(validate_data,  batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data)
    
    # train_loader
    
    dataloaders = [train_loader, validate_loader, test_loader]
    return train_loader, validate_loader, test_loader, train_data

def main():
    # Loading data
    trainloader, validloader, testloader, train_data = loadData(where)

#     # Setting up network
#     model, criterion = fmodel.setup_network(struct, dropout, hidden_units, lr, power)
#     optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model, optimizer, criterion = nn_measure()
    # Training loop
    steps = 0
    running_loss = 0
    print_every = 3
    print("--Training starting--")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1

            if torch.cuda.is_available() and power == 'gpu':
                inputs, labels = inputs.to(device), labels.to(device)
                model = model.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Validation
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    # Save checkpoint
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({
        'structure': struct,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': lr,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, path)
    print("Saved checkpoint!")

if __name__ == "__main__":
    main()