import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from PIL import Image
import numpy as np

def getcommandlineargs():
    parser = argparse.ArgumentParser(description='Image Classifier App')
    parser.add_argument('data_dir', type=str, help='Valid location of data directory')
    parser.add_argument('--save_dir', action="store", dest="save_dir")
    parser.add_argument('--arch', action="store", dest="arch")
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int)
    parser.add_argument('--epochs', action="store", dest="epochs", type=int)
    parser.add_argument('--gpu', action="store_true", default=False)
    return parser.parse_args()

def getdataloader(train_dir, valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    return trainloader, validloader, train_data

def getmodelinputsize(nn_model, arch):
    if arch.startswith("vgg"):
        return nn_model.classifier[0].in_features
    elif arch.startswith("densenet"):
        return nn_model.classifier.in_features
    elif arch == 'alexnet':
        return nn_model.classifier[1].in_features
    elif arch.startswith("resnet") or arch.startswith("inception"):
        return nn_model.fc.in_features

def buildmodel(arch, hidden_units):
    if arch == None:
        arch='vgg16'
    if hidden_units == None:
        hidden_units = 256
        
    model =  getattr(models,arch)(pretrained=True)
    input_size = getmodelinputsize(model,arch)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('input', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.5)),
                              ('fc', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    return model
    
def save_checkpoint(model, train_data, directory_path, arch):
    checkpoint = {'arch' : arch,
              'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx}
    if directory_path == None:
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, directory_path+'/checkpoint.pth')
    
def trainmodel(model,optimizer,trainloader,validloader,device,epochs):
    steps = 0
    running_loss = 0
    print_every = 10
    total_start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()                
                validation_loss += batch_loss.item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
        print("---Epoch %s trained in %s seconds ---" % (epoch+1, (time.time() - epoch_start_time)))
    print("--- %s seconds ---" % (time.time() - total_start_time))
    return model

if __name__ == "__main__":
    c_args = getcommandlineargs()
    data_dir = c_args.data_dir
    isGPU = c_args.gpu
    arch = c_args.arch
    save_dir = c_args.save_dir
    learning_rate = c_args.learning_rate
    hidden_units = c_args.hidden_units
    epochs = c_args.epochs
    
    
    train_directory = data_dir + '/train'
    valid_directory = data_dir + '/valid'
    trainloader, validloader, train_data = getdataloader(train_directory, valid_directory)
    
    device = "cpu"
    optimizer = None
    if isGPU:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise Exception("Cuda device is not activated !")
    print("Current active device :", device)
    model = buildmodel(arch, hidden_units)
    criterion = nn.NLLLoss()
    if learning_rate:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    model.to(device)
    print('---Model Training Started---')
    model = trainmodel(model,optimizer,trainloader,validloader,device,epochs)
    print('---Model Trained Successfully---')
    save_checkpoint(model,train_data,save_dir,arch)
    print('---Model Saved---')
    