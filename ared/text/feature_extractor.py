import os
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments, GPT2DoubleHeadsModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW



class TextFeatureExtractor:
    """
    Extracts text features using a pre-trained GPT-2 model.
    
    The `TextFeatureExtractor` class loads a pre-trained GPT-2 model and tokenizer, and provides a method to extract features from a given utterance. The extracted features are the hidden states from the last layer of the GPT-2 model.
    
    Args:
        model_chekpoint (str): The path to the pre-trained GPT-2 model checkpoint.
        device (str, optional): The device to use for the model, either 'cuda' or 'cpu'. Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: The extracted features for the given utterance.
    """
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
    
    def extract_features(self, utterance: str) -> torch.Tensor:
        """
        Extracts features from the given utterance using the pre-trained GPT-2 model.
        
        Args:
            utterance (str): The input utterance to extract features from.
        
        Returns:
            torch.Tensor: The extracted features for the given utterance.
        """
        inp = self.tokenizer(utterance, return_tensors='pt', padding=True, truncation=True)
        inp.to(self.device)
        out = self.model(**inp, output_hidden_states=True)
        encoded_utterance = out['hidden_states'][12][0]
        return encoded_utterance