import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

class ASR:
    def __init__(self, device='cuda'):
        torch.manual_seed(1234)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio", device_map=self.device, trust_remote_code=True).eval()

    def convert_speech_to_text(self, file_path: str):
        if not os.path.exists(file_path):
            raise ValueError("Invalid file")
        
        file_extension = file_path.split('.')[-1]
        
        if file_extension not in ['mp4', 'avi', 'mp3', 'wav', 'flac']:
            raise ValueError("Unsupported file format, only accepts mp4, avi, mp4 wav and flac files")
        
        sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
        query = f"<audio>{file_path}</audio>{sp_prompt}"
        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors='pt', audio_info=audio_info)
        inputs = inputs.to(self.device)
        pred = self.model.generate(**inputs, audio_info=audio_info)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False,audio_info=audio_info)
        return response