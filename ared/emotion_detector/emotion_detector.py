from typing import Union
import numpy as np
from ared.audio import AudioPreprocessor
from ared.vision import VidePreprocessor
from ared.text import TextPreprocessor
from ared.models import MMER
from ared.utils import (
    NameSpace, id2emotion
    )
from PIL import Image

class EmotionDetector:

    HYPER_PARAMS = NameSpace(
        {
            'out_dim': 7,
            'embracenet_feat_dim': 256,
            'orig_d_mod_l': 512,
            'orig_d_mod_v': 768,
            'orig_d_mod_a': 256,
            'conv_dim': 30,
            'num_heads': 5,
            'layers': 5,
            'attn_dropout': 0.1,
            'relu_dropout': 0.1,
            'res_dropout': 0.25,
            'embed_dropout': 0.1,
            'attn_mask': True,
            'device': 'cuda',
            'epochs': 100,
            'batch_size': 16,
            'lr': 1e-5,
            'model_name': 'MELD-MMER',
        }
    )
    
    def __init__(self, 
                vis_model_weights: str, 
                text_model_weights: str, 
                audio_model_weights: str, 
                device: str='cuda'):
        
        self.HYPER_PARAMS.device = device
        self.model = MMER(self.HYPER_PARAMS).to(device)
        self.model.eval()
        
        self.vision = VidePreprocessor(vision_extractor_weights=vis_model_weights, device=device)
        self.audio = AudioPreprocessor(model_weights=audio_model_weights, device=device)
        self.text = TextPreprocessor(model_weights=text_model_weights, device=device)
        
    def detect_emotion(self, audio, video, text):
    
        audio = self.compute_audio_features(audio)
        video = self.compute_vision_features(video)
        text = self.text.process(text)
        
        output_probs = self.model(audio.unsqueeze(0), 
                            video.unsqueeze(0), 
                            text.unsqueeze(0))
        
        emotion, output_probs = self._process_output(output_probs)

        return (emotion, output_probs)
    
    def compute_vision_features(self, video):
        if isinstance(video, str):
            video = self.vision.process_video(video)
        elif isinstance(video, list):
            # TODO add a fix to check if the object is of type PIL Image
            # if not isinstance(video[0], Image):
            #     raise ValueError("provide video as either list of PIL images or as a string path to the video")    
            video = self.vision.process_images(video)
        else:
            raise ValueError("provide video as either list of PIL images or as a string path to the video")        
        return video
    
    def compute_audio_features(self, audio: Union[str,np.ndarray]):
        if isinstance(audio, str):
            file_extension = audio.split('.')[-1]
            if file_extension not in ['mp4', 'mp3', 'avi', 'wav']:
                raise ValueError("audio features can only be extracted form an mp3, mp4, avi, and wav files only")
        
        audio = self.audio.process_audio_file(audio)
        return audio
    
    def _process_output(self, model_output):
        emotion = id2emotion[model_output.argmax().cpu().item()]
        output_probs = {id2emotion[i]: model_output.cpu()[0][i] for i in range(len(id2emotion))}
        return emotion, output_probs
    
        