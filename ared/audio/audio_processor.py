from .feature_extractor import AudioFeatureExtractor
from moviepy.editor import VideoFileClip, AudioClip
from ared.utils import timeit

class AudioPreprocessor:
    
    def __init__(self, model_weights: str):
        self.audio_feature_extractor = AudioFeatureExtractor(model_weights)

        self.audio_features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.audio_features[name] = output[0].detach()
            return hook
        self.audio_feature_extractor.model.wave_rnn.register_forward_hook(get_activation('wave_rnn'))
    
    @timeit
    def process_video_file(self, video_file: str):
        v_clip = VideoFileClip(video_file)
        a_clip = v_clip.audio
        a_clip.write_audiofile('./temp.wav')
        v_clip.close()
        return self.process_audio_file('./temp.wav')
    
    @timeit
    def process_audio_file(self, audio_file: str):
        self.audio_feature_extractor.extract_features(audio_file)
        return self.audio_features['wave_rnn'][0][-50:, :]
        