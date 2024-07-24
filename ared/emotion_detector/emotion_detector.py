from typing import Union
import numpy as np
import torch
from ared.audio import AudioPreprocessor
from ared.vision import VidePreprocessor
from ared.text import TextPreprocessor
from ared.models import MMER
from ared.utils import (
    NameSpace, id2emotion
    )
from PIL import Image
from typing import Union

class EmotionDetector:
    """
    The EmotionDetector class is responsible for detecting emotions from audio, video, and text inputs. 
    It uses a pre-trained MMER (Multimodal Emotion Recognition) model to perform the emotion detection.

    The class has the following methods:

    - `__init__(self, vis_model_weights: str, text_model_weights: str, audio_model_weights: str, device: str='cuda')`: Initializes the EmotionDetector with the specified model weights and device.
    - `detect_emotion(self, audio, video, text)`: Detects the emotion from the given audio, video, and text inputs. Returns the detected emotion and the output probabilities for each emotion.
    - `compute_vision_features(self, video)`: Computes the vision features from the given video input, which can be either a string path to a video file or a list of PIL Image objects.
    - `compute_audio_features(self, audio: Union[str,np.ndarray])`: Computes the audio features from the given audio input, which can be either a string path to an audio file or a numpy array.
    - `_process_output(self, model_output)`: Processes the output of the MMER model to extract the detected emotion and the output probabilities for each emotion.
    """

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
        """
        Initializes the EmotionDetector class with the specified model weights and device.
        
        Args:
            vis_model_weights (str): Path to the pre-trained vision model weights.
            text_model_weights (str): Path to the pre-trained text model weights.
            audio_model_weights (str): Path to the pre-trained audio model weights.
            device (str, optional): The device to use for the models. Defaults to 'cuda'.
        
        Attributes:
            HYPER_PARAMS (NameSpace): A namespace containing the hyperparameters for the MMER model.
            model (MMER): The pre-trained MMER model.
            vision (VidePreprocessor): The vision preprocessor.
            audio (AudioPreprocessor): The audio preprocessor.
            text (TextPreprocessor): The text preprocessor.
        """
                
        self.HYPER_PARAMS.device = device
        self.model = MMER(self.HYPER_PARAMS).to(device)
        self.model.eval()
        
        self.vision = VidePreprocessor(vision_extractor_weights=vis_model_weights, device=device)
        self.audio = AudioPreprocessor(model_weights=audio_model_weights, device=device)
        self.text = TextPreprocessor(model_weights=text_model_weights, device=device)
        
    def detect_emotion(self, audio, video, text):
        """
        Detects the emotion from the given audio, video, and text inputs.
        
        Args:
            audio (Union[str, np.ndarray]): The audio input, either as a file path or a numpy array.
            video (Union[str, List[PIL.Image.Image]]): The video input, either as a file path or a list of PIL Image objects.
            text (str): The text input.
        
        Returns:
            Tuple[str, Dict[str, float]]: A tuple containing the detected emotion and a dictionary of output probabilities for each emotion.
        """
        
        audio = self.compute_audio_features(audio)
        video = self.compute_vision_features(video)
        text = self.text.process(text)
        
        output_probs = self.model(
            audio.unsqueeze(0), 
            video.unsqueeze(0), 
            text.unsqueeze(0)
            )
        
        emotion, output_probs = self._process_output(output_probs)

        return (emotion, output_probs)
    
    def compute_vision_features(self, video: Union[str, list[Image.Image]]):
        """
        Computes the vision features from the given video input.
        
        Args:
            video (Union[str, List[PIL.Image.Image]]): The video input, either as a file path or a list of PIL Image objects.
        
        Returns:
            torch.Tensor: The computed vision features.
        
        Raises:
            ValueError: If the video input is not a valid file path or a list of PIL Image objects.
        """
        
        if isinstance(video, list):
            if not isinstance(video[0], Image.Image):
                raise ValueError("provide video as either list of PIL images or a string path to the video")    
        
        video = self.vision.process_images(video)      
        return video
    
    def compute_audio_features(self, audio: Union[str,np.ndarray]):
        """
        Computes the audio features from the given audio input.
        
        Args:
            audio (Union[str, np.ndarray]): The audio input, either as a file path or a numpy array.
        
        Returns:
            torch.Tensor: The computed audio features.
        
        Raises:
            ValueError: If the audio input is not a valid file path for an mp3, mp4, avi, or wav file.
        """
        if isinstance(audio, str):
            file_extension = audio.split('.')[-1]
            if file_extension not in ['mp4', 'mp3', 'avi', 'wav']:
                raise ValueError("audio features can only be extracted form an mp3, mp4, avi, and wav files only")
        
        audio = self.audio.process_audio_file(audio)
        return audio
    
    def _process_output(self, model_output: torch.Tensor):
        """
        Processes the output of the emotion detection model.
        
        Args:
            model_output (torch.Tensor): The output tensor from the emotion detection model.
        
        Returns:
            Tuple[str, Dict[str, float]]: A tuple containing the predicted emotion and a dictionary of output probabilities for each emotion.
        """
        emotion = id2emotion[model_output.argmax().cpu().item()]
        output_probs = {id2emotion[i]: model_output.cpu()[0][i] for i in range(len(id2emotion))}
        return emotion, output_probs
    
        