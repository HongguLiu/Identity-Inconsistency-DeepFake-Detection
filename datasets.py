from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import albumentations as A
import cv2
import random
import torch

class MyDataset(Dataset):
    def __init__(self, root='/nas/home/hliu/Datasets/FF++/', transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
        fh.close()
        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, 1-label

class VideoDataset(Dataset):
    def __init__(self, txt_path, sequence_length = 20, transform=None, types='Deepfakes', quality='raw'):
        fh = open(txt_path, 'r')
        videos = []
        types_list = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
        for video in fh:
            video = video.rstrip()
            video = video.replace('quality', quality)
            if types == 'All' and not 'Origin' in video:
                for tp in types_list:
                    video = video.replace('type', tp)
                    videos.append(video)
            else:
                video = video.replace('type', types)
                videos.append(video)
        fh.close()
        self.videos = videos
        self.transform = transform
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        v_path = self.videos[index]
        frames = os.listdir(v_path)
        frames.sort()
        start_max = len(frames)-self.sequence_length
        if start_max <= 0:
            print(v_path)
            v_path = '/nas/home/hliu/Datasets/FF++/Origin/raw/000/'
            frames = os.listdir(v_path)
            frames.sort()
            start_max = len(frames)-self.sequence_length
        start = np.random.randint(0,start_max)
        imgs=[]
        for i in range(start, start+self.sequence_length):
            fn = os.path.join(v_path, frames[i])
            img = Image.open(fn).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                imgs.append(img)
            else:
                imgs.append(img)
        imgs = torch.stack(imgs)
        if 'Origin' in v_path:
            label = 0  #real is 0
        else:
            label = 1  #fake is 1
        return imgs, label

class VideoDataset_test(Dataset):
    def __init__(self, txt_path, sequence_length = 20, transform=None, types='Deepfakes', quality='raw'):
        fh = open(txt_path, 'r')
        videos = []
        for video in fh:
            video = video.rstrip()
            video = video.replace('type', types)
            video = video.replace('quality', quality)
            videos.append(video)
        fh.close()
        self.videos = videos
        self.transform = transform
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        v_path = self.videos[index]
        frames = os.listdir(v_path)
        frames.sort()
        seg = len(frames) // self.sequence_length
        frames = frames[0:seg*self.sequence_length]
        # start_max = len(frames)-self.sequence_length
        # if start_max <= 0:
        #     print(v_path)
        #     v_path = '/nas/home/hliu/Datasets/FF++/Origin/raw/000/'
        #     frames = os.listdir(v_path)
        #     frames.sort()
        #     start_max = len(frames)-self.sequence_length
        # start = np.random.randint(0,start_max)
        imgs=[]
        # for i in range(start, start+self.sequence_length):
        for i in range(len(frames)):
            fn = os.path.join(v_path, frames[i])
            img = Image.open(fn).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
                imgs.append(img)
            else:
                imgs.append(img)
        imgs = torch.stack(imgs)
        if 'Origin' in v_path:
            label = 0  #real is 0
        else:
            label = 1  #fake is 1
        return imgs, label, seg

class VideoDataset_aug(Dataset):
    def __init__(self, txt_path, sequence_length = 20, transform=None, types='Deepfakes', quality='raw'):
        fh = open(txt_path, 'r')
        videos = []
        for video in fh:
            video = video.rstrip()
            video = video.replace('type', types)
            video = video.replace('quality', quality)
            videos.append(video)
        fh.close()
        self.videos = videos
        self.transform = transform
        self.sequence_length = sequence_length
        aug = A.Compose([
            A.ChannelShuffle(p=0.2),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            A.OneOf([
                A.CoarseDropout(max_height=4, max_width=4, p=0.5),
                A.PixelDropout(p=0.1)
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),            
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        self.aug = aug
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        v_path = self.videos[index]
        frames = os.listdir(v_path)
        frames.sort()
        start_max = len(frames)-self.sequence_length
        if start_max <= 0:
            print(v_path)
            v_path = '/nas/home/hliu/Datasets/FF++/Origin/raw/000/'
            frames = os.listdir(v_path)
            frames.sort()
            start_max = len(frames)-self.sequence_length
        start = np.random.randint(0,start_max)
        imgs=[]
        for i in range(start, start+self.sequence_length):
            fn = os.path.join(v_path, frames[i])
            img = Image.open(fn).convert('RGB')
            img_np = np.array(img)
            img = self.aug(image=img_np)["image"]
            if self.transform is not None:
                img = self.transform(img)
                imgs.append(img)
            else:
                imgs.append(img)
        imgs = torch.stack(imgs)
        if 'Origin' in v_path:
            label = 0  #real is 0
        else:
            label = 1  #fake is 1
        return imgs, label