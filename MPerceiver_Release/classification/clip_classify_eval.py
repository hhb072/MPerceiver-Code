from turtle import forward
import torch
from PIL import Image
import open_clip
from glob import glob
from tqdm import tqdm
from torch import nn
import os 
import random
import math 
from PIL import Image
import numpy as np 
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch import nn 
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

class Dataset_eval(Dataset):
    def __init__(self,root_path,transform=None):
        self.root_path = root_path
        self.transform = transform
        self.files = os.listdir(root_path)
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.root_path,self.files[idx]))
        if self.transform:
            img = self.transform(img)
        return img
        


class ImageClassifier(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        model,_,_ = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
        self.model = model.visual
        self.fc1 = nn.Linear(1024,512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512,class_num)

    def freeze(self):
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def freeze_clip(self):
        for name, param in self.named_parameters():
            if 'model' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self,image):
        return self.fc2(self.relu(self.fc1(self.model(image))))

# model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device('cpu'), pretrained="laion2b_s32b_b79k")
# # model = model.cuda()
# # image = preprocess()
# # print(model.visual.image_size)
# image = preprocess(Image.open('512.png')).unsqueeze(0)
# classfi = ImageClassifier(8)
# image_feature = classfi(image)
# print(image_feature.shape)
# print(preprocess)

def _convert_to_rgb(image):
    return image.convert('RGB')


if __name__ == '__main__':
    # x = Image.open('/data2/yuang.ai/data/all_in_one/raindrop/RainDrop/train/data/99.png')
    train_transforms = transforms.Compose([transforms.Resize((224,224),
    interpolation=InterpolationMode.BICUBIC),
    _convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    model = ImageClassifier(9)
    model.load_state_dict(torch.load('/data2/yuang.ai/code/Airfusion/classification/epoch=32.pth',map_location=torch.device('cpu')))
    model.eval()
    model.cuda()
    path = '/data3/yuang.ai/code/daclip-uir-main/datasets/universal/val/snowy/LQ'
    label = 1
    dataset = Dataset_eval(path,train_transforms)
    dataloader = DataLoader(dataset,batch_size=32,shuffle=False,num_workers=32)
    correct = 0
    total = 0
    for i,data in tqdm(enumerate(dataloader)):
        image = data.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        total += image.size(0)
        correct += (predicted == label).sum().item()
        print(correct/total)
    print(correct/total)

    # model = ImageClassifier(9)
    # model.freeze_clip()
    # total = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    # print("Number of parameter: %.4fM" % (total / 1e6))

