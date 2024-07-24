
from typing import Union
import numpy as np
import torch
from torchaudio.models import WaveRNN
import torchaudio
from torchaudio.transforms import MelSpectrogram
from ared.utils import timeit
import librosa as lb

class WaveRNNClassification(torch.nn.Module):
    def __init__(self, upsample_scales=[5,5,8], n_classes=3, kernel_size=5, hop_length=200):
        super(WaveRNNClassification, self).__init__()
        self.n_classes = n_classes
        
        self.wave_rnn = WaveRNN(upsample_scales=upsample_scales, 
                                n_classes=n_classes, 
                                kernel_size=kernel_size, 
                                hop_length=hop_length)
        
        self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, waveform, spec):
        x = self.wave_rnn(waveform, spec)
        x = self.softmax(x)
        x = x[:, -1, self.n_classes]
        return x
    
    
class AudioFeatureExtractor:
    def __init__(self, model_weights_path: str, device: str='cuda'):
        self.rnn_kernel_size=5
        self.device = device
        
        self.model = WaveRNNClassification(upsample_scales=[5,5,8], n_classes=3, kernel_size=5, hop_length=200)
        self.model.state_dict = torch.load(model_weights_path)

        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 44100
        self.n_fft=400
        self.win_length=400
        self.hop_length=200
        self.power=2.0
        self.norm=None
        
    def extract_features(self, audio_data_or_file: Union[str, np.ndarray]):
        if isinstance(audio_data_or_file, str):
            file_extension = audio_data_or_file.split('.')[-1]
            
            if file_extension not in ['mp4', 'avi', 'wav', 'mp3', 'flac']:
                raise ValueError("Unsupported file, only support mp4 avi wav mp3 and flac formats for audio")
            
            waveform, sample_rate = torchaudio.load(audio_data_or_file)
            spec = MelSpectrogram(sample_rate)(waveform)
            
        elif isinstance(audio_data_or_file, np.ndarray):
            waveform = audio_data_or_file
            spec = lb.feature.melspectrogram(y=waveform, 
                                             sr=self.sample_rate,
                                             n_fft=self.n_fft,
                                             win_length=self.win_length,
                                             hop_length=self.hop_length,
                                             power=self.power,
                                             norm=self.norm)
            waveform = torch.tensor(waveform)
            spec = torch.tensor(spec)
        
        print("waveform shape: ", waveform.shape)
        print("Spec shape: ", spec.shape)

        # should do mean over channels
        if waveform.shape[0] == 2:
            waveform = waveform[0, :]
            spec = spec[0, :]
        waveform = waveform.unsqueeze(0)
        spec = spec.unsqueeze(0)
        _, _, n_time = spec.shape
        waveform = waveform[:, :(n_time-self.rnn_kernel_size +1)*self.hop_length]
        
        # make the prediction
        waveform = waveform.to(self.device).unsqueeze(0)
        spec = spec.to(self.device).unsqueeze(0)
       
        encoded_audio = self.model(waveform, spec)       
        return encoded_audio