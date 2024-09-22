import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop

import cv2
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])

def transform():
    return Compose([
        ToTensor(),
        
    ])
    
def load_img(filepath):
    img = Image.open(filepath)
    return img



class Data_eval(data.Dataset):#self, config, is_train=True,
    def __init__(self,cfg, transform=transform()):
        super(Data_eval, self).__init__()
        self.cfg = cfg
        data_dir_A = cfg[cfg['test_dataset']]['data_dir']['data_dir_A']
        data_dir_B = cfg[cfg['test_dataset']]['data_dir']['data_dir_B']
        self.A_image_filenames = sorted([join(data_dir_A, x) for x in listdir(data_dir_A) if is_image_file(x)])
        self.B_image_filenames = sorted([join(data_dir_B, x) for x in listdir(data_dir_B) if is_image_file(x)])
      
        self.transform = transform


    def __getitem__(self, index):

        A_image = load_img(self.A_image_filenames[index])
        B_image = load_img(self.B_image_filenames[index])

        _, file = os.path.split(self.A_image_filenames[index])
            
        if self.transform:
            real_A = self.transform(A_image)
            real_B = self.transform(B_image)
       
        return real_A,real_B,file

        
        
    def __len__(self):
        return len(self.A_image_filenames)