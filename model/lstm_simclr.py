import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_simclr(nn.Module):
    def __init__(self, num_classes=2, latent_dim=512, lstm_layers=3, hidden_dim=512, sequence_length=20, dropout=0, bidirectional=False, batch_first=True):
        super(LSTM_model, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=batch_first, drop_out=dropout, bidirectional=bidirectional)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes))
        # self.bn = nn.BatchNorm1d(latent_dim)


    # x: batch_size, sequence_length, latent_dim
    def forward(self, x):
        # add flatten_parameters()
        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x, None)
        # x = self.linear(torch.mean(x_lstm, dim=1))
        x = self.mlp(x_lstm)
        # x = self.relu(x)
        # x = self.dropout(x)
        return x


import timm

class SpatialNet(nn.Module):
    def __init__(self, name="xception", num_classes=2):
        super(SpatialNet, self).__init__()
        self.net = timm.create_model(model_name=name, pretrained=True, num_classes=num_classes)
        self.spatial = nn.Sequential(*list(self.net.children())[:-1])
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b*s, c, h, w)
        out = self.spatial(x)
        out = torch.mean(out, dim=0).unsqueeze(dim=0)
        # out.shape: [b,2048]
        return out

class LSTM_model(nn.Module):
    '''
    20221212,add batch_first = True
    '''
    def __init__(self, num_classes=2, latent_dim=512, lstm_layers=3, hidden_dim=512, sequence_length=20, dropout=0, bidirectional=False, batch_first=True):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = lstm_layers
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, batch_first=batch_first)
        self.relu = nn.LeakyReLU()

    # x: batch_size, sequence_length, 25088
    def forward(self, x):
        # add flatten_parameters()
        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x)
        x = x_lstm[:,-1,:]
        x = self.relu(x)
        return x

class IDC(nn.Module):
    def __init__(self, sequence_length, num_classes=2):
        super(IDC, self).__init__()
        self.s_model = SpatialNet(name="xception", num_classes=num_classes)
        self.t_model = LSTM_model(num_classes=num_classes, latent_dim=25088, lstm_layers=3, hidden_dim=2048, sequence_length=sequence_length)
        # self.att = nn.MultiheadAttention(2048*2, 8)
        self.mlp = nn.Sequential(nn.Linear(2048*2, 2048*2), nn.ReLU(), nn.Linear(2048*2, num_classes))
        # self.fc = nn.Linear(2048, num_classes)
    def forward(self, x, id_feature):
        out_s = self.s_model(x)
        out_t = self.t_model(id_feature)
        features = torch.cat((out_s, out_t), dim=1)
        # output, attention_weights = self.att(features, features, features)
        # output = self.fc(out_s)
        output = self.mlp(features)
        return output
