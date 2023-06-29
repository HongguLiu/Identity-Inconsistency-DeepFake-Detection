import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
    
    def forward(self, x):
        output, hidden = self.lstm(x)
        return output, hidden

class VideoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(VideoEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
    
    def forward(self, x):
        output, hidden = self.lstm(x)
        return output, hidden

class AudioVideoTransformer(nn.Module):
    def __init__(self, audio_hidden_size, video_hidden_size, num_heads, num_layers):
        super(AudioVideoTransformer, self).__init__()
        
        self.audio_encoder = AudioEncoder(input_size=80, hidden_size=audio_hidden_size)
        self.video_encoder = VideoEncoder(input_size=2048, hidden_size=video_hidden_size)
        
        self.transformer = nn.Transformer(d_model=audio_hidden_size+video_hidden_size, 
                                           nhead=num_heads, num_encoder_layers=num_layers, 
                                           num_decoder_layers=num_layers, dim_feedforward=512, dropout=0.1)
        
        self.fc = nn.Linear(audio_hidden_size+video_hidden_size, 1)
    
    def forward(self, audio_input, video_input):
        audio_output, _ = self.audio_encoder(audio_input)
        video_output, _ = self.video_encoder(video_input)
        
        x = torch.cat([audio_output, video_output], dim=0)
        
        x = self.transformer(x, x)
        
        x = self.fc(x)
        
        return x
