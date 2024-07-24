import torch
from .feature_extractor import TextFeatureExtractor

class TextPreprocessor:
    """
    Preprocesses text data by extracting features using a TextFeatureExtractor.
    
    Args:
        model_weights (str): The path to the model weights for the TextFeatureExtractor.
        device (str, optional): The device to use for the TextFeatureExtractor. Defaults to 'cuda'.
    
    Returns:
        The extracted features for the input text.
    """
        
    def __init__(self, model_weights: str, device: str='cuda'):
        self.device = device
        self.text_feature_extractor = TextFeatureExtractor(model_weights, device=self.device)
    
    def process(self, text: str) -> torch.Tensor:
        return self.text_feature_extractor.extract_features(text)