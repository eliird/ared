from .feature_extractor import AudioFeatureExtractor
from moviepy.editor import VideoFileClip, AudioClip
from ared.utils import timeit
import numpy as np
from typing import Union

class AudioPreprocessor:
    """
    Preprocesses audio data by extracting features using a pre-trained model.
    
    Args:
        model_weights (str): Path to the pre-trained model weights.
        device (str, optional): The device to use for the model, either 'cuda' or 'cpu'. Defaults to 'cuda'.
    
    Attributes:
        device (str): The device used for the model.
        audio_feature_extractor (AudioFeatureExtractor): The feature extractor used to process audio files.
        audio_features (dict): A dictionary to store the extracted audio features.
    
    Methods:
        process_audio_file(audio_file):
            Extracts features from the given audio file and returns the last 50 feature vectors.
    """
        
    def __init__(self, model_weights: str, device: str='cuda'):
        '''
        Initializes the AudioPreprocessor class with the specified model weights and device. 
        Sets up a hook to capture the output of the wave_rnn layer in the audio feature extractor model, 
        which will be stored in the audio_features dictionary.
        '''
        self.device = device
        self.audio_feature_extractor = AudioFeatureExtractor(model_weights, device=self.device)

        self.audio_features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.audio_features[name] = output[0].detach()
            return hook
        self.audio_feature_extractor.model.wave_rnn.fc2.register_forward_hook(get_activation('wave_rnn'))
    
    def process_audio_file(self, audio_file: Union[str, np.ndarray]):
        '''
        Extracts the last 50 layers of the pretrianed WaveRNN model and returns them as a tensor.
        Args:
            audio_file (str|np.ndarray): The path to the audio file to process.
        
        Returns:
            torch.Tensor: The last 50 feature vectors extracted from the audio data.
        '''
        self.audio_feature_extractor.extract_features(audio_file)
        return self.audio_features['wave_rnn'][-50:, :]
        