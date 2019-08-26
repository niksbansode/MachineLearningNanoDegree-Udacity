import argparse
import torch
import json
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import warnings
    
def getpredictargs():
    parser = argparse.ArgumentParser(description='Image Classifier App')
    parser.add_argument('image_path', type=str, help='Valid image file path')
    parser.add_argument('checkpoint_path', type=str, help='Valid checkpoint file path')
    parser.add_argument('--top_k', action="store", dest="top_k", type=int)
    parser.add_argument('--category_names', action="store", dest="cat_to_name_file")
    parser.add_argument('--gpu', action="store_true", default=False)
    return parser.parse_args()

def load_checkpoint(file_path, gpu=False):
    checkpoint = None
    if gpu:
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def loadcatjson(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_image(image):
    image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])
    
    PIL_image = Image.open(image)
    PIL_image = image_transforms(PIL_image)
    np_image = np.array(PIL_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image.transpose((1, 2, 0)) - mean)/std    
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def printresult(probs, classes):
    df = pd.DataFrame({'Class': classes, 'Probability': probs})
    print('************************************')
    print('---Prediction---')
    print(df.head(1))
    print('************************************')
    print('---Top ',len(classes),' most likely classes---')
    print(df)
    print('************************************')

def predict(image_path, loaded_model, topk, cat, is_gpu):
    probs = None
    labels = None
    if topk == None:
        topk=5
    if is_gpu:
        loaded_model = loaded_model.to(device)
        input_torch_img = input_torch_img.to(device)
    else:
        loaded_model.cpu()
    numpy_img = process_image(image_path)
    input_torch_img = torch.from_numpy(numpy_img).unsqueeze(0).float()
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model.forward(input_torch_img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        if is_gpu:
            probs = top_p.cpu().numpy().tolist()[0]
            labels = top_class.cpu().numpy().tolist()[0]
        else:
            probs = top_p.numpy().tolist()[0]
            labels = top_class.numpy().tolist()[0]  
        
        probs = top_p.numpy().tolist()[0]
        labels = top_class.numpy().tolist()[0]
        index_class_dict = {val: key for key, val in 
                        loaded_model.class_to_idx.items()}
        labels = [index_class_dict[label] for label in labels]
        top_flowers = [cat[label] for label in labels]
        return probs, top_flowers
    
if __name__ == "__main__":
    # Ignore deprecation warnings
    warnings.filterwarnings("ignore")
    
    c_args = getpredictargs()
    image_path = c_args.image_path
    checkpoint_path = c_args.checkpoint_path
    top_k = c_args.top_k
    cat_to_name_file = c_args.cat_to_name_file
    is_gpu_enabled = c_args.gpu
    
    loaded_model = load_checkpoint(checkpoint_path, is_gpu_enabled)
    cat_to_name = loadcatjson(cat_to_name_file)
    probs, names = predict(image_path, loaded_model, top_k, cat_to_name, is_gpu_enabled)
    printresult(probs, names)