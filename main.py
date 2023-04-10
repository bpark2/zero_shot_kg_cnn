import gc
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from PIL import Image
from torch.autograd import Variable

'''
Pytorch tutorial followed:
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- https://pytorch.org/hub/pytorch_vision_resnet/
'''

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
torch.manual_seed(42)
model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

image_vectors = {}
data_dir = "snake_images"
for s_class in os.listdir(data_dir+"/train_data"):
    image_vectors["train_data"] = {s_class:[]}
    image_vectors["test_data"] = {s_class:[]}
num_classes = len(os.listdir(data_dir+"/train_data"))
batch_size = 8
epochs = 100
lr = 0.001
feature_extract = True
set_parameter_requires_grad(model, feature_extract)
model.fc = nn.Linear(model.fc.in_features, num_classes)
input_size = 64

data_transforms = {
    'train_data': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_data': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train_data', 'test_data']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train_data', 'test_data']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
params = model.parameters()

#if we're feature extracting, only update certain parameters
if feature_extract:
    params_to_up = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_up.append(param)


optim = torch.optim.Adam(params_to_up,lr=lr)
crit = nn.CrossEntropyLoss()

#print("Cuda Available: ",torch.cuda.is_available())

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_data', 'test_data']:
            if phase == 'train_data':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                with torch.set_grad_enabled(phase == 'train_data'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train_data':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data) #modify this if we want top-1 accuracy

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test_data' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test_data':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def blind_eval(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Beginning Evaluation")#print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders["test_data"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)  # modify this if we want top-1 accuracy

            epoch_loss = running_loss / len(dataloaders["test_data"].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders["test_data"].dataset)

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format("test_data", epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)

        time_elapsed = time.time() - since
        print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

def get_vectors(image):
    model.eval()
    layer = model._modules.get('avgpool')
    img = Image.open(image).convert('RGB')
    trans_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).to(device)
    emb = torch.zeros(2048)
    def copy_data(m, i, o):
        emb.copy_(o.data.reshape(o.data.size(1)))
    h = layer.register_forward_hook(copy_data)
    model(trans_img)
    h.remove()
    return emb.tolist()

    #print(model)

if __name__ == '__main__':
    model, hist = train_model(model,dataloaders_dict, crit,optim, num_epochs=epochs, is_inception=False)
    torch.save(model,"resnet_fine_tuned.pkl")
    #uncomment above for 100 epochs of fine-tuning

    #uncomment below for evaluation using no fine-tuning
    # model, hist = blind_eval(model,dataloaders_dict, crit,optim, num_epochs=epochs, is_inception=False)
    # torch.save(model,"resnet_zero_shot.pkl")

    #uncomment below for getting image vectors from model (not fine-tuned) pretrained on imagenet
    # scaler = transforms.Resize((224, 224))
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # to_tensor = transforms.ToTensor()
    # for phase in ['train_data','test_data']:
    #     for species in os.listdir(path=data_dir+"/"+phase):
    #         for image in os.listdir(data_dir+"/"+phase+"/"+species):
    #             image_vectors[phase][species] = get_vectors(data_dir+"/"+phase+"/"+species+"/"+image)
    # with open('image_vectors.pkl','wb') as fp:
    #     pickle.dump(image_vectors,fp)
    #     print("dict saved")