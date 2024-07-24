from .feature_extractor import AudioFeatureExtractor
from moviepy.editor import VideoFileClip, AudioClip
from ared.utils import timeit

class AudioPreprocessor:
    
    def __init__(self, model_weights: str, device: str='cuda'):
        self.device = device
        self.audio_feature_extractor = AudioFeatureExtractor(model_weights, device=self.device)

        self.audio_features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.audio_features[name] = output[0].detach()
            return hook
        self.audio_feature_extractor.model.wave_rnn.fc2.register_forward_hook(get_activation('wave_rnn'))
    
    def process_audio_file(self, audio_file):
        self.audio_feature_extractor.extract_features(audio_file)
        return self.audio_features['wave_rnn'][-50:, :]
        