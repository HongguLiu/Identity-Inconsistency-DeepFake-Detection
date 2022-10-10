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

from datasets import MyDataset, VideoDataset, VideoDataset_aug, VideoDataset_test

from model.base_model import Identity_model, LSTM_model, get_model

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from torchvision import transforms

import sys


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

def train_epoch(epoch, num_epochs, data_loader, model_id, model_lstm, criterion, optimizer, log_path=None):
    print("*********Training is begin*********")
    model_id.eval() #Identity_model for inference purposes
    model_lstm.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (inputs, targets) in enumerate(data_loader):
        # print(targets)
        if torch.cuda.is_available():
            targets = targets.type(torch.cuda.LongTensor)
            inputs = inputs.cuda()
            model_id.cuda()
            model_lstm.cuda()
        feature_id = model_id(inputs)
        id_feature = feature_id.detach() # detach the feature_id from the model_id.
        optimizer.zero_grad()
        outputs = model_lstm(id_feature)
        # print(outputs)
        loss  = criterion(outputs, targets.type(torch.cuda.LongTensor))
        acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print_log("[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.4f%%]"% (epoch, num_epochs, i, len(data_loader), losses.avg, accuracies.avg), log_path)
    print_log('[Epoch {} / {}] Training Accuracy: {}'.format(epoch, args.epochs, accuracies.avg), log_path)

def test(epoch, data_loader, model_id, model_lstm, criterion, test=True, log_path=None):
    if test:
        print('Testing.......')
    model_id.eval()
    model_lstm.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
                model_id.cuda()
                model_lstm.cuda()
            feature_id = model_id(inputs)
            id_feature = feature_id.detach() # detach the feature_id from the model_id.
            outputs = model_lstm(id_feature)
            acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
            _, p = torch.max(outputs,1) 
            true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            accuracies.update(acc, inputs.size(0))
        print_log('[Epoch {} / {}] Accuracy {}'.format(epoch, args.epochs, accuracies.avg), log_path)
    return true, pred, accuracies.avg

types = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'All']
qualities = ['raw', 'c23', 'c40']

im_size = 112
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ID DeepFake Detection Training')
    parser.add_argument('--name', type=str, required=True, default=None, help='The name of the experiment')
    parser.add_argument('--network', type=str, default='r50', help='Arcface backbone network')
    parser.add_argument('--weight', type=str, default='/root/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth', help='Arcface pretrained weight')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-8, help='Weight decay rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=1, help='Number of Training')
    parser.add_argument('--num_classes', '-n', type=int, default=2, help='Number of Classes')
    parser.add_argument('--latent_dim', '-ld', type=int, default=25088, help='Number of Latent Dimensions')
    parser.add_argument('--num_layers', '-nl', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=2048, help='Number of Hidden Dimensions')
    parser.add_argument('--sequence_length', '-sq', type=int, default=20, help='Number of Sequence Lengths')
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--train_file', type=str, default='/nas/home/hliu/Datasets/FF++/data_list/ffpp_train.txt', help='The file list of training examples')
    parser.add_argument('--val_file', type=str, default='/nas/home/hliu/Datasets/FF++/data_list/ffpp_val.txt', help='The file list of validation examples')
    parser.add_argument('--test_file', type=str, default='/nas/home/hliu/Datasets/FF++/data_list/ffpp_test.txt', help='The file list of test examples')
    parser.add_argument('--type', type=str, required=True, help='The type of Fake video, you should choose from types')
    parser.add_argument('--quality', type=str, default='raw', help='The quality of the video, you should choose from qualities')
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--balance_weight', type=bool, default=False, help='Balance the real and fake.')



    args = parser.parse_args()
    if args.type not in types or args.quality not in qualities:
        raise ValueError('Invalid type: %s' % args.type)

    output_path = os.path.join('/nas/home/hliu/fakeid_detection/training', args.name, 'checkpoints')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join('/nas/home/hliu/fakeid_detection/training', args.name) + '/logs.txt'

    cmd = sys.argv
    print_log(" ".join(cmd), log_path)
    
    model_id = Identity_model(args.network, args.weight)
    # import pdb
    # pdb.set_trace()
    model_lstm = LSTM_model(args.num_classes, args.latent_dim, args.num_layers, args.hidden_dim, args.sequence_length, args.bidirectional)

    model_id = nn.DataParallel(model_id)
    model_lstm = nn.DataParallel(model_lstm)

    train_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.Normalize(mean,std)])
    
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.Normalize(mean,std)])
    
    if args.aug:
        print_log("Use augumentation for training.....", log_path)
        train_dataset = VideoDataset_aug(args.train_file, args.sequence_length, train_transforms, args.type, args.quality)
    else:
        train_dataset = VideoDataset(args.train_file, args.sequence_length, train_transforms, args.type, args.quality)
    val_dataset = VideoDataset(args.val_file, args.sequence_length, test_transforms, args.type, args.quality)
    test_dataset = VideoDataset(args.test_file, args.sequence_length, test_transforms, args.type, args.quality)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    if args.balance_weight:
        class_weights = torch.from_numpy(np.asarray([1,0.2])).type(torch.FloatTensor).cuda()
        criterion = nn.CrossEntropyLoss(weight = class_weights).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()


    optimizer = torch.optim.Adam(model_lstm.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)
    # optimizer = torch.optim.SGD(model_lstm.parameters(), lr = args.learning_rate, weight_decay = args.weight_decay)

    best_val_epoch = 0
    best_test_epoch = 0
    best_val_auc = 0
    best_test_auc = 0

    for epoch in range(1, args.epochs+1):
        train_epoch(epoch, args.epochs, train_loader, model_id, model_lstm, criterion, optimizer, log_path=log_path)
        v_true, v_pred, v_acc = test(epoch, val_loader, model_id, model_lstm, criterion, test=False, log_path=log_path) #validate the model
        v_auc = roc_auc_score(v_true, v_pred)
        print_log("The Validation accuracy is:{:.4f}\nAUC is: {:.4f}\n".format(v_acc, v_auc), log_path)
        print_log(classification_report(v_true, v_pred, labels=[0, 1], target_names=['Real', args.type]), log_path)
        if v_auc > best_val_auc:
            best_val_epoch = epoch
            best_val_auc = v_auc
            best_val_model = model_lstm.module.state_dict()

        t_true, t_pred, t_acc = test(epoch, test_loader, model_id, model_lstm, criterion, log_path=log_path)
        t_auc = roc_auc_score(t_true, t_pred)
        print_log("The Test accuracy is:{:.4f}\nAUC is: {:.4f}\n".format(t_acc, t_auc), log_path)
        print_log(classification_report(t_true, t_pred, labels=[0, 1], target_names=['Real', args.type]), log_path)
        if t_auc > best_test_auc:
            best_test_epoch = epoch
            best_test_auc = t_auc
            best_test_model = model_lstm.module.state_dict()
        if (epoch+1) % (int(args.epochs/5)) == 0:
            torch.save(model_lstm.module.state_dict(), os.path.join(output_path, str(epoch)+'_'+str(v_auc)[0:4]+"val_"+str(t_auc)[0:4]+"test.pt"))
            print_log('Save the {} model of {:.4f}val_{:.4f}test.pt'.format(epoch, v_auc, t_auc), log_path)
        
    torch.save(best_val_model, os.path.join(output_path, str(best_val_epoch)+'_'+str(best_val_auc)[0:6]+"best_val.pt"))
    torch.save(best_test_model, os.path.join(output_path, str(best_test_epoch)+'_'+str(best_test_auc)[0:6]+"best_test.pt"))
