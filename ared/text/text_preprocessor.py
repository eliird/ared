from .feature_extractor import TextFeatureExtractor

class TextPreprocessor:
    
    def __init__(self, model_weights: str, device: str='cuda'):
        self.device = device
        self.text_feature_extractor = TextFeatureExtractor(model_weights, device=self.device)
    
    def process(self, text: str):
        return self.text_feature_extractor.extract_features(text)