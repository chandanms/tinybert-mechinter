import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TinyBertModel:
    def __init__(self, model_name='prajjwal1/bert-tiny'):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_layers = len(self.model.encoder.layer)
        self.num_heads = self.model.config.num_attention_heads
    
    def inference(self, text, output_attentions=True):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs, output_attentions=output_attentions)
        print (outputs)
        

    def print_model_info(self):
        """Print the complete model architecture"""
        print("\nModel Architecture:")
        print(f"Number of layers: {self.num_layers}")
        print(f"Number of attention heads: {self.num_heads}")
        print("\nLayer components:")
        for name, module in self.model.named_modules():
            if len(name) > 0:  # Skip empty name (root module)
                print(f"- {name}")