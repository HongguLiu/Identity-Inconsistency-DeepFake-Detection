import argparse
import os
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

# from torch.utils.tensorboard import SummaryWriter

from datasets import MyDataset, VideoDataset_spatial, VideoDataset_test_spatial

from model.base_model import Identity_model, get_model

from model.lstm_simclr import IDC, SpatialNet, LSTM_model

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from torchvision import transforms

import sys

import pdb



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size

def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # write logs into log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')

types = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'All', 'celebdfv2']
qualities = ['raw', 'c23', 'c40']

im_size = 112
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

def test(epoch, data_loader, model_id, IDC_net, criterion, test=True, log_path=None):
    if test:
        print('Testing.......')
    model_id.eval()
    IDC_net.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    with torch.no_grad():
        for i, (inputs, inputs_s, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                targets = targets.type(torch.cuda.LongTensor)
                inputs = inputs.cuda()
                inputs_s = inputs_s.cuda()
                model_id.cuda()
                IDC_net.cuda()
            feature_id = model_id(inputs)
            id_feature = feature_id.detach() # detach the feature_id from the model_id.
            # outputs = lstm_id(id_feature)
            # import pdb
            # pdb.set_trace()
            outputs = IDC_net(inputs_s, id_feature)
            # outputs = IDC_net(inputs_s, inputs_s)
            acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
            _, p = torch.max(outputs,1) 
            true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            accuracies.update(acc, inputs.size(0))
        print_log('[Epoch {} / {}] Accuracy {}'.format(epoch, args.epochs, accuracies.avg), log_path)
    return true, pred, accuracies.avg

def test_allframes(data_loader, model_id, IDC_net, criterion, test=True, log_path=None):
    if test:
        print('Testing.......')
    model_id.eval()
    IDC_net.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    prob = []
    true = []
    with torch.no_grad():
        for i, (inputs, inputs_s, targets, seg) in enumerate(data_loader):
            torch.cuda.empty_cache()
            seg = seg.item()
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
                inputs_s = inputs_s.cuda()
                model_id.cuda()
                IDC_net.cuda()
            feature_id = model_id(inputs)
            id_feature = feature_id.detach() # detach the feature_id from the model_id.
            # the shape of id_feature is B * S * 25088 
            b, s, _ = id_feature.shape
            n_correct_segs = 0
            fake_cnt = 0
            for j in range(seg):
                torch.cuda.empty_cache()
                inputs_s_seg = inputs_s[:,j*args.sequence_length:(j+1)*args.sequence_length,:]
                id_feature_seg = id_feature[:,j*args.sequence_length:(j+1)*args.sequence_length,:]
                # print(id_feature_seg.shape)
                output_seg = IDC_net(inputs_s, id_feature_seg)
                _, p_seg = torch.max(output_seg, 1)
                batch_size = targets.size(0)
                p_seg_t = p_seg.t()#  p_seg tensor([0])
                if p_seg_t.item() == 1:
                    fake_cnt += 1
                correct = p_seg_t.eq(targets.view(1, -1))
                n_correct_segs += correct.float().sum().item()
            
            prob.append(fake_cnt / seg)
            acc = n_correct_segs / seg
            pred.append(int((fake_cnt / seg)>=0.5))


            # outputs = model_lstm(id_feature)
            # acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
            # _, p = torch.max(outputs,1) 
            true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            # pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            accuracies.update(acc, inputs.size(0))
            # import pdb
            # pdb.set_trace()
            # print("Accuracy:{:.4f}\tProb:{:.4f}\tPred:{:.4f}\n".format(acc, fake_cnt / seg, int((fake_cnt / seg)>=0.5)))
        # print_log('Test Accuracy {}'.format(accuracies.avg), log_path)
    return true, pred, prob, accuracies.avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ID DeepFake Detection Training')
    parser.add_argument('--name', type=str, required=True, default=None, help='The name of the experiment')
    parser.add_argument('--network', type=str, default='r50', help='Arcface backbone network')
    parser.add_argument('--weight', type=str, default='/root/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth', help='Arcface pretrained weight')
    parser.add_argument('--checkpoints', '-c', type=str, required=True, help='The checkpoints of LSTM.')
    parser.add_argument('--batch_size', '-bs', type=int, default=1, help='Number of Training')
    parser.add_argument('--num_classes', '-n', type=int, default=2, help='Number of Classes')
    parser.add_argument('--latent_dim', '-ld', type=int, default=25088, help='Number of Latent Dimensions')
    parser.add_argument('--num_layers', '-nl', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=2048, help='Number of Hidden Dimensions')
    parser.add_argument('--sequence_length', '-sq', type=int, default=20, help='Number of Sequence Lengths')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--test_file', type=str, default='/nas/home/hliu/Datasets/FF++/data_list/ffpp_test.txt', help='The file list of test examples')
    parser.add_argument('--type', type=str, required=True, help='The type of Fake video, you should choose from types')
    parser.add_argument('--quality', type=str, default='raw', help='The quality of the video, you should choose from qualities')



    args = parser.parse_args()
    if args.type not in types or args.quality not in qualities:
        raise ValueError('Invalid type: %s' % args.type)
    
    model_id = Identity_model(args.network, args.weight)
    IDC_net = IDC(sequence_length=args.sequence_length, num_classes=args.num_classes)

    # model_id = nn.DataParallel(model_id)
    # IDC_net = nn.DataParallel(IDC_net)

    
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.Normalize(mean,std)])
    
    
    test_transforms_spatial = transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize((299,299)),
                                                transforms.Normalize(mean,std)])
    


    test_dataset = VideoDataset_test_spatial(args.test_file, args.sequence_length, test_transforms, test_transforms_spatial, args.type, args.quality)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss().cuda()


    IDC_net.load_state_dict(torch.load(args.checkpoints))



    

    if not os.path.exists(os.path.join('/nas/home/hliu/fakeid_detection/test_2023', args.name)):
        os.makedirs(os.path.join('/nas/home/hliu/fakeid_detection/test_2023', args.name))

    log_path = os.path.join('/nas/home/hliu/fakeid_detection/test_2023', args.name) + '/test_reports.txt'

    cmd = sys.argv
    print_log(" ".join(cmd), log_path)

    # t_true, t_pred, t_acc = test(test_loader, model_id, model_lstm, criterion, log_path=log_path)
    t_true, t_pred, t_prob, t_acc = test_allframes(test_loader, model_id, IDC_net, criterion, log_path=log_path)
    t_auc = roc_auc_score(t_true, t_pred)
    t_auc_prob = roc_auc_score(t_true, t_prob)
    # print("The Test accuracy is:{:.4f}\nAUC is: {:.4f}\n".format(t_acc, t_auc))
    print_log("The Test pred accuracy is:{:.4f}\n".format(t_acc), log_path)
    print_log("The Test prob AUC is: {:.4f}\nThe pred AUC is: {:.4f}".format(t_auc_prob, t_auc), log_path)
    # print_log(classification_report(t_true, t_pred, target_names=[args.type, 'real']), log_path)
    print_log(classification_report(t_true, t_pred, target_names=['Real', args.type]), log_path) # 0 is real, 1 is fake



    
