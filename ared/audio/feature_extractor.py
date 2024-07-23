

import torch
from torchaudio.models import WaveRNN
import torchaudio
from torchaudio.transforms import MelSpectrogram


class WaveRNNClassification(torch.nn.Module):
    def __init__(self, upsample_scales=[5,5,8], n_classes=3, kernel_size=5, hop_length=200):
        super(WaveRNNClassification, self).__init__()
        self.n_classes = n_classes
        self.wave_rnn = WaveRNN(upsample_scales=upsample_scales, n_classes=n_classes, kernel_size=kernel_size, hop_length=hop_length)
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self, waveform, spec):
        x = self.wave_rnn(waveform, spec)
        x = self.softmax(x)
        x = x[:, -1, self.n_classes]
        return x
    
    
class AudioFeatureExtractor:
    def __init__(self, model_weights_path: str):
        self.rnn_kernel_size=5
        self.n_hops = 200
        self.device = 'cuda'
        
        
        self.model = WaveRNNClassification(upsample_scales=[5,5,8], n_classes=3, kernel_size=5, hop_length=200)
        self.model.state_dict = torch.load(model_weights_path)

        self.model.to(self.device)
        self.model.eval()
         
    def extract_features(self, audio_data_or_file):
        waveform, sample_rate = torchaudio.load(audio_data_or_file)
        spec = MelSpectrogram(sample_rate)(waveform)
        
        # should do mean over channels
        waveform = waveform[0, :].unsqueeze(0)
        spec = spec[0, :].unsqueeze(0)
        
        _, _, n_time = spec.shape
        waveform = waveform[:, :(n_time-self.rnn_kernel_size +1)*self.n_hops]
        
        # make the prediction
        waveform = waveform.to(self.device).unsqueeze(0)
        spec = spec.to(self.device).unsqueeze(0)
       
        encoded_audio = self.model(waveform, spec)       
        return encoded_audio