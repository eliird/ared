from ared.audio import AudioPreprocessor
from ared.vision import VidePreprocessor
from ared.text import TextPreprocessor


class EmotionDetector:
    def __init__(self, model, vis_processor, text_processor, audio_processor):
        self.model = model
        self.vision = vis_processor
        self.audio = audio_processor
        self.text = text_processor
        
    
    def detect_emotion(self, audio, video, text):
        audio = self.audio(audio)
        video - self.vision(video)
        text = self.text(text)
        
        output = self.model(audio, video, text)
        output = self.process_output(output)
        return output
    
    
    def _process_output(self, model_output):
        pass
    
        