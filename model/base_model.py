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
        # self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=batch_first, drop_out=dropout, bidirectional=bidirectional)
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
        feature_id = feature_id.view(b, s, 512*7*7)
        return feature_id


class LSTM_model(nn.Module):
    '''
    20221212,add batch_first = True
    '''
    def __init__(self, num_classes=2, latent_dim=512, lstm_layers=3, hidden_dim=512, sequence_length=20, dropout=0, bidirectional=False, batch_first=True):
        super(LSTM_model, self).__init__()
        # bias = False
        # self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional, batch_first=batch_first)

        # bias = True and no bn after 20/12/22
        # bn and bias after 2022/01/25
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=lstm_layers, bias=True, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        '''
        初始化权重
        '''
        # for name, param in self.lstm.named_parameters():
            # nn.init.uniform_(param,-0.1,0.1)
        self.relu = nn.LeakyReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(512,num_classes)
        self.linear = nn.Linear(hidden_dim,num_classes)
        # self.linear_model = nn.Linear(25088, 2048, bias=False)
        # self.bn = nn.BatchNorm1d(20)
        # self.bn = nn.BatchNorm1d(sequence_length)

    # x: batch_size, sequence_length, 25088
    def forward(self, x):
        # add flatten_parameters()
        self.lstm.flatten_parameters()
        # b,s,_ = x.shape
        # x = x.view(b, s*25088)
        # x = self.bn(x)
        # x = x.view(b, s, 25088)
        # x  = self.linear_model(x)
        x_lstm, _ = self.lstm(x, None)
        # x = self.relu(x)
        # x = self.linear(torch.mean(x_lstm, dim=1))
        x = self.linear(x_lstm[:,-1,:])
        x = self.relu(x)
        # x = self.dropout(x)
        return x

class Arcface_model(nn.Module):
    def __init__(self, name, weight):
        super(Arcface_model, self).__init__()
        net = get_model(name, fp16=False)
        weight = '/nas/home/hliu/insightface/model_zoo/arcface_torch/ms1mv3_arcface_r50_fp16/backbone.pth'
        net.load_state_dict(torch.load(weight))
        # self.model = net
        self.model = nn.Sequential(*list(net.children())[:-2])
        self.model = net
        # self.linear_model = nn.Linear(25088, 2048, bias=False)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b*s, c, h, w)
        feature_id = self.model(x) # bs * 512
        # feature_id = torch.flatten(feature_id, 1)
        feature_id = feature_id.view(b, s, 512)
        return feature_id