
from typing import Union
import numpy as np
import torch
from torchaudio.models import WaveRNN
import torchaudio
from torchaudio.transforms import MelSpectrogram
from ared.utils import timeit
import librosa as lb


class WaveRNNClassification(torch.nn.Module):
    """
    Defines a PyTorch module for classifying audio waveforms using a WaveRNN model.
    
    The `WaveRNNClassification` module inherits from `torch.nn.Module` and is used to classify audio waveforms into one of `n_classes` classes.
    It does this by passing the waveform and a spectrogram representation of the audio through a `WaveRNN` model, 
    and then applying a softmax activation to the output to get the class probabilities.
    
    The module has the following parameters:
    - `upsample_scales`: A list of integers specifying the upsampling scales for the WaveRNN model.
    - `n_classes`: The number of classes to classify the audio into.
    - `kernel_size`: The kernel size for the WaveRNN model.
    - `hop_length`: The hop length for the WaveRNN model.
    
    The `forward` method takes a waveform tensor and a spectrogram tensor as input, and returns a tensor of class probabilities.
    """
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
    """
    Defines an `AudioFeatureExtractor` class that can extract audio features from either a file path or a NumPy array of audio data.
    
    The `AudioFeatureExtractor` class has the following methods:
    
    - `__init__(self, model_weights_path: str, device: str='cuda')`: Initializes the `AudioFeatureExtractor` with the specified model weights path and device (default is 'cuda').
    - `extract_features(self, audio_data_or_file: Union[str, np.ndarray])`: Extracts audio features from either a file path or a NumPy array of audio data. If a file path is provided, it supports the following audio formats: mp4, avi, wav, mp3, and flac. The extracted features are returned as a tensor.
    """
    def __init__(self, model_weights_path: str, device: str='cuda'):
        """
        Initializes an `AudioFeatureExtractor` instance with the specified model weights path and device.
        
        The `__init__` method sets up the necessary attributes and loads the pre-trained model weights onto the specified device. 
        It also sets up the default parameters for the audio feature extraction, such as sample rate, FFT window size, and hop length.
        
        Args:
            model_weights_path (str): The file path to the pre-trained model weights.
            device (str, optional): The device to load the model onto, defaults to 'cuda'.
        """
        self.rnn_kernel_size=5
        self.device = device
        
        self.model = WaveRNNClassification(upsample_scales=[5,5,8], n_classes=3, kernel_size=5, hop_length=200)
        self.model.state_dict = torch.load(model_weights_path, map_location=self.device)

        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = 44100
        self.n_fft=400
        self.win_length=400
        self.hop_length=200
        self.power=2.0
        self.norm=None
        
    def extract_features(self, audio_data_or_file: Union[str, np.ndarray]):
        """
        Extracts audio features from either a file path or a NumPy array of audio data.
        
        If a file path is provided, it supports the following audio formats: mp4, avi, wav, mp3, and flac. 
        The extracted features are returned as a tensor.
        
        Args:
            audio_data_or_file (Union[str, np.ndarray]): Either a file path or a NumPy array of audio data.
        
        Returns:
            torch.Tensor: The extracted audio features.
        """
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