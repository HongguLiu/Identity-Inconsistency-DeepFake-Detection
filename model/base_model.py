from .__init__ import get_model
import torch
import torch.nn as nn
import torch.nn.functional as F

# name = 'r50'
# weight = '/nas/home/hliu/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth'
# net = get_model(name, fp16=False)
# net.load_state_dict(torch.load(weight))

class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=512, lstm_layers=3, hidden_dim = 512, bidirectional = False, name='r50', weight=None):
        super(Model, self).__init__()
        name = 'r50'
        weight = '/nas/home/hliu/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth'
        net = get_model(name, fp16=False)
        net.load_state_dict(torch.load(weight))
        # self.model = nn.Sequential(*list(net.children())[:-2])
        # self.linear_model = nn.Linear(25088, 2048, bias=False)
        self.model = net
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512,num_classes)
    
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b*s, c, h, w)
        feature_id = self.model(x)
        # print(feature_id.shape)
        # x = self.linear_model(feature_id)
        x = feature_id.view(b, s, 512)
        x_lstm, _ = self.lstm(x, None)
        x = self.linear(torch.mean(x_lstm, dim=1))
        x = self.relu(x)
        x = self.dropout(x)
        return feature_id, x

class Identity_model(nn.Module):
    def __init__(self, name, weight):
        super(Identity_model, self).__init__()
        net = get_model(name, fp16=False)
        weight = '/nas/home/hliu/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth'
        net.load_state_dict(torch.load(weight))
        # self.model = net
        self.model = nn.Sequential(*list(net.children())[:-2])
        # self.linear_model = nn.Linear(25088, 2048, bias=False)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b*s, c, h, w)
        feature_id = self.model(x) # 512 * 7 * 7
        feature_id = torch.flatten(feature_id, 1)
        feature_id = feature_id.view(b, s, 25088)
        # feature_id = self.model(x)
        # feature_id = feature_id.view(b, s, 512)
        return feature_id


class LSTM_model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=512, lstm_layers=3, hidden_dim=512, sequence_length=20, bidirectional=False):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(512,num_classes)
        self.linear = nn.Linear(hidden_dim,num_classes)
        # self.linear_model = nn.Linear(25088, 2048, bias=False)
        # self.bn = nn.BatchNorm1d(20)
        self.bn = nn.BatchNorm1d(sequence_length)

    def forward(self, x):
        # b,s,_ = x.shape
        # x = x.view(b, s*25088)
        x = self.bn(x)
        # x = x.view(b, s, 25088)
        # x  = self.linear_model(x)
        x_lstm, _ = self.lstm(x, None)
        x = self.linear(torch.mean(x_lstm, dim=1))
        x = self.relu(x)
        # x = self.dropout(x)
        return x