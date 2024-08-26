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
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import argparse


#Modified from WGWS-Net, Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions (CVPR'23)
class AIO_Dataset(Dataset):
    def __init__(self, rootA_in, rootB_in, rootC_in,rootD_in,rootE_in,rootF_in,rootG_in,rootH_in,rootI_in,
                fix_sample_A = 500,fix_sample_B =500 ,fix_sample_C =500, fix_sample_D = 500,fix_sample_E = 500,fix_sample_F = 500,fix_sample_G = 500,fix_sample_H = 500,fix_sample_I = 500,
                transform=None):
        self.fix_sample_A = fix_sample_A
        in_files_A = os.listdir(rootA_in)
        if self.fix_sample_A > len(in_files_A):
            self.fix_sample_A = len(in_files_A)
        in_files_A = random.sample(in_files_A, self.fix_sample_A)   
        self.imgs_in_A = [os.path.join(rootA_in, k) for k in in_files_A]
        len_imgs_in_A = len(self.imgs_in_A)
        self.length = len_imgs_in_A
        self.r_l_rate = 1 
        self.r_l_rate1 = 1  

        in_files_B = os.listdir(rootB_in)
        self.fix_sample_B = fix_sample_B
        if self.fix_sample_B >len(in_files_B):
            self.fix_sample_B = len(in_files_B)
        in_files_B = random.sample(in_files_B, self.fix_sample_B)
        self.imgs_in_B = [os.path.join(rootB_in, k) for k in in_files_B]

        len_imgs_in_B_ori = len(self.imgs_in_B )
        self.imgs_in_B = self.imgs_in_B * (self.r_l_rate + math.ceil(len_imgs_in_A / len_imgs_in_B_ori))  
        self.imgs_in_B = self.imgs_in_B[0: self.r_l_rate * len_imgs_in_A]

        in_files_C = os.listdir(rootC_in)
        self.fix_sample_C = fix_sample_C
        if self.fix_sample_C > len(in_files_C):
            self.fix_sample_C = len(in_files_C)
        in_files_C = random.sample(in_files_C, self.fix_sample_C)
        self.imgs_in_C = [os.path.join(rootC_in, k) for k in in_files_C]

        len_imgs_in_C_ori = len(self.imgs_in_C)  
        self.imgs_in_C = self.imgs_in_C * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_C_ori))  
        self.imgs_in_C = self.imgs_in_C[0: self.r_l_rate1 * len_imgs_in_A]

        in_filesD = os.listdir(rootD_in)
        self.fix_sample_D = fix_sample_D
        if self.fix_sample_D > len(in_filesD):
            self.fix_sample_D = len(in_filesD)
        in_filesD = random.sample(in_filesD, self.fix_sample_D)
        self.imgs_in_D = [os.path.join(rootD_in, k) for k in in_filesD]

        len_imgs_in_D_ori = len(self.imgs_in_D) 
        self.imgs_in_D = self.imgs_in_D * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_D_ori))  
        self.imgs_in_D = self.imgs_in_D[0: self.r_l_rate1 * len_imgs_in_A]

        in_filesE = os.listdir(rootE_in)
        self.fix_sample_E = fix_sample_E
        if self.fix_sample_E > len(in_filesE):
            self.fix_sample_E = len(in_filesE)
        in_filesE = random.sample(in_filesE, self.fix_sample_E)
        self.imgs_in_E = [os.path.join(rootE_in, k) for k in in_filesE]

        len_imgs_in_E_ori = len(self.imgs_in_E) 
        self.imgs_in_E = self.imgs_in_E * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_E_ori)) 
        self.imgs_in_E = self.imgs_in_E[0: self.r_l_rate1 * len_imgs_in_A]

        in_filesF = os.listdir(rootF_in)
        self.fix_sample_F = fix_sample_F
        if self.fix_sample_F > len(in_filesF):
            self.fix_sample_F = len(in_filesF)
        in_filesF = random.sample(in_filesF, self.fix_sample_F)
        self.imgs_in_F = [os.path.join(rootF_in, k) for k in in_filesF]

        len_imgs_in_F_ori = len(self.imgs_in_F)  
        self.imgs_in_F = self.imgs_in_F * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_F_ori)) 
        self.imgs_in_F = self.imgs_in_F[0: self.r_l_rate1 * len_imgs_in_A]

        in_filesG = os.listdir(rootG_in)
        self.fix_sample_G = fix_sample_G
        if self.fix_sample_G > len(in_filesG):
            self.fix_sample_G = len(in_filesG)
        in_filesG = random.sample(in_filesG, self.fix_sample_G)
        self.imgs_in_G = [os.path.join(rootG_in, k) for k in in_filesG]

        len_imgs_in_G_ori = len(self.imgs_in_G)  
        self.imgs_in_G = self.imgs_in_G * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_G_ori))  
        self.imgs_in_G = self.imgs_in_G[0: self.r_l_rate1 * len_imgs_in_A]

        in_filesH = os.listdir(rootH_in)
        self.fix_sample_H = fix_sample_H
        if self.fix_sample_H > len(in_filesH):
            self.fix_sample_H = len(in_filesH)
        in_filesH = random.sample(in_filesH, self.fix_sample_H)
        self.imgs_in_H = [os.path.join(rootH_in, k) for k in in_filesH]

        len_imgs_in_H_ori = len(self.imgs_in_H)  
        self.imgs_in_H = self.imgs_in_H * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_H_ori))  
        self.imgs_in_H = self.imgs_in_H[0: self.r_l_rate1 * len_imgs_in_A]

        in_filesI = os.listdir(rootI_in)
        self.fix_sample_I = fix_sample_I
        if self.fix_sample_I > len(in_filesI):
            self.fix_sample_I = len(in_filesI)
        in_filesI = random.sample(in_filesI, self.fix_sample_I)
        self.imgs_in_I = [os.path.join(rootI_in, k) for k in in_filesI]

        len_imgs_in_I_ori = len(self.imgs_in_I)  
        self.imgs_in_I = self.imgs_in_I * (self.r_l_rate1 + math.ceil(len_imgs_in_A / len_imgs_in_I_ori))  
        self.imgs_in_I = self.imgs_in_I[0: self.r_l_rate1 * len_imgs_in_A]
        
        self.transform = transform

    def __len__(self):
        return len(self.imgs_in_A)

    def __getitem__(self, idx):
        data_A = self.transform(Image.open(self.imgs_in_A[idx]))
        data_B = self.transform(Image.open(self.imgs_in_B[idx]))
        # data_C = self.transform(Image.open(self.imgs_in_C[idx]))

        data_C = np.array(Image.open(self.imgs_in_C[idx])) #index=2 refers to Gaussian Noise
        sigma = np.random.choice([25])
        data_C = np.clip(data_C+np.random.normal(0,sigma,data_C.shape),0,255)
        data_C = Image.fromarray(data_C.astype(np.uint8))
        data_C = self.transform(data_C)
        
        data_D = self.transform(Image.open(self.imgs_in_D[idx]))
        data_E = self.transform(Image.open(self.imgs_in_E[idx]))
        data_F = self.transform(Image.open(self.imgs_in_F[idx]))
        data_G = self.transform(Image.open(self.imgs_in_G[idx]))
        data_H = self.transform(Image.open(self.imgs_in_H[idx]))
        data_I = self.transform(Image.open(self.imgs_in_I[idx]))

        return [data_A,0],[data_B,1],[data_C,2],[data_D,3],[data_E,4],[data_F,5],[data_G,6],[data_H,7],[data_I,8]


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
                print(name+'will be updated!')
    
    def forward(self,image):
        return self.fc2(self.relu(self.fc1(self.model(image))))

def _convert_to_rgb(image):
    return image.convert('RGB')


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction='mean'):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  
        log_softmax = torch.log_softmax(pred, dim=1) 
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  
        logpt = logpt.view(-1) 
        ce_loss = -logpt  
        pt = torch.exp(logpt) 
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss 
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
        
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '1234'      
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    rootA_in = '/data2/yuang.ai/data/aio_9/haze/RESIDE/train/input' 
    rootB_in = '/data2/yuang.ai/data/aio_9/snow/snow100k/train/input' 
    rootC_in = '/data3/yuang.ai/code/daclip-uir-main/datasets/universal/train/noisy/GT' 
    rootD_in = '/data2/yuang.ai/data/all_in_one/rain/Rain1400/rainy_image_dataset/training/rainy_image'
    rootE_in = '/data3/yuang.ai/code/daclip-uir-main/datasets/universal/train/motion-blurry/LQ'
    rootF_in = '/data3/yuang.ai/code/daclip-uir-main/datasets/universal/train/defocus-blurry/LQ'
    rootG_in = '/data3/yuang.ai/code/daclip-uir-main/datasets/universal/train/real-noisy/LQ'
    rootH_in = '/data2/yuang.ai/data/aio_9/raindrop/raindrop/train/input'
    rootI_in = '/data3/yuang.ai/code/daclip-uir-main/datasets/universal/train/low-light/LQ'

    fix_sample = 100000
    train_transforms = transforms.Compose([transforms.Resize((224,224),
    interpolation=InterpolationMode.BICUBIC),
    _convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    train_set = AIO_Dataset(rootA_in,rootB_in,rootC_in,rootD_in,rootE_in,rootF_in,rootG_in,rootH_in,rootI_in,
                          fix_sample,fix_sample,fix_sample,fix_sample,fix_sample,fix_sample,fix_sample,fix_sample,fix_sample,
                          train_transforms)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_set, batch_size=192, shuffle=False, num_workers=4, sampler=train_sampler)
    model = ImageClassifier(9)
    model.freeze_clip()
    model = model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = torch.nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    num_epoch = 100

    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, train_data in enumerate(train_loader):
            data_A, data_B, data_C, data_D,data_E,data_F,data_G,data_H,data_I = train_data
            data_in = torch.cat([data_A[0],data_B[0],data_C[0],data_D[0],data_E[0],data_F[0],data_G[0],data_H[0],data_I[0]],dim=0)
            label = torch.cat([data_A[1],data_B[1],data_C[1],data_D[1],data_E[1],data_F[1],data_G[1],data_H[1],data_I[1]],dim=0)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs = Variable(data_in).to(f'cuda:{rank}')
            labels = Variable(label).to(f'cuda:{rank}')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:    
                if rank == 0: 
                    print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        scheduler.step()

        if rank == 0:
            torch.save(model.module.state_dict(), f'epoch={epoch}.pth')
    cleanup()
if __name__ == '__main__':
    world_size = 4
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)