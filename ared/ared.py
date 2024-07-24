from ared.emotion_detector import EmotionDetector
from ared.models import MMER
from ared.utils import (
    id2emotion, emotion2id, load_first_50_images
)
import random

random.seed(20)

def build_emotion_detector(vision_weights: str, 
                audio_weights: str, 
                text_weights: str, 
                device: str='cuda'):
    
    detector = EmotionDetector(
        vis_model_weights = vision_weights, 
        text_model_weights = text_weights, 
        audio_model_weights = audio_weights,
        device = device
        )
    
    return detector