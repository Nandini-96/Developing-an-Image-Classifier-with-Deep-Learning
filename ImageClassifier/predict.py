
## Import json and argparse
import json
import argparse
from torch import nn, optim
## Import torch, numpy and PIL
import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import numpy as np
import PIL
from PIL import Image
import torch.nn.functional as F

from math import ceil
from torchvision import models


arg = argparse.ArgumentParser(
    description='predict-file')
arg.add_argument('input_img', default='./flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
arg.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
arg.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
arg.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
arg.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = arg.parse_args()
json_name = pa.category_names
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint

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
    return  train_data

def load_checkpoint(path='checkpoint.pth'):
    model = torchvision.models.vgg16(pretrained=True)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    train_dataset=loadData();
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
        }
    torch.save(checkpoint, 'checkpoint.pth')
    checkpoint = torch.load(path)
      # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
   
    
    # TODO: Process a PIL image for use in a PyTorch model
    image_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
                                                                
#     pil_image = Image.open(image)
    tensor_image = image_transforms(image)
    return tensor_image

def predict(image_path, model, topk,cat_to_name):  
    model.to('cuda')
    
    img=Image.open(image_path)
    img_t = process_image(img)
    img_t = img_t.unsqueeze_(0)
    img_t = img_t.float()
    # Move input tensor to the GPU
    img_t = img_t.to('cuda')
    with torch.no_grad():
        ps = torch.exp(model(img_t))
        
    ps, top_classes = ps.topk(topk, dim=1)
#     ps= ps.cpu().detach().numpy().tolist()[0]
#     top_classes = top_classes.cpu().detach().numpy().tolist()[0]
    idx_to_flower = {v:cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] if i in idx_to_flower else "Unknown" for i in top_classes.tolist()[0]]

    # returning both as lists instead of torch objects for simplicity
    return ps.tolist()[0], predicted_flowers_list


model = load_checkpoint()

with open(pa.category_names, 'r') as f:
        cat_to_name = json.load(f)
top_ps, top_classes = predict(path_image, model, pa.top_k, cat_to_name)
print("Predictions:")
for i in range(pa.top_k):
    print("#{: <3} {: <10} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i]*100))

