import os
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, GPT2DoubleHeadsModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW



class TextFeatureExtractor:
    def __init__(self, model_chekpoint: str, device: str='cuda'):
        model_name = "gpt2"

        self.device = device        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Add padding token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new [PAD] token
        
        self.model = GPT2DoubleHeadsModel.from_pretrained(model_chekpoint).to(device) # , num_labels=7 3 for your 3 labels
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer)) 
        self.model.eval()
    
    def extract_features(self, utterance):
        inp = self.tokenizer(utterance, return_tensors='pt', padding=True, truncation=True)
        inp.to(self.device)
        out = self.model(**inp, output_hidden_states=True)
        encoded_utterance = out['hidden_states'][12][0]
        return encoded_utterance