import argparse
import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

from datasets from MyDataset

from .backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    # print(feat)
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ID DeepFake Detection Training')
    parser.add_argument('--network', type=str, default='r50', help='Arcface backbone network')
    parser.add_argument('--weight', type=str, default='/root/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth', help='Arcface pretrained weight')
    # parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    # import pdb
    # pdb.set_trace()
    # feat = inference(args.weight, args.network, args.img)
    # pdb.set_trace()