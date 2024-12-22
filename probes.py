from torch import nn
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

class AttentionProbe(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.model = bert_model
        self.num_layers = bert_model.num_layers
        self.num_heads = bert_model.num_heads

        self.head_size = bert_model.model.config.hidden_size // bert_model.model.config.num_attention_heads
        
        # Hook storage
        self.attention_with_values = []
        self.handles = []
        
        # Register hooks for each layer
        for layer_idx in range(self.num_layers):
            layer = self.model.model.encoder.layer[layer_idx].attention.self
            handle = layer.register_forward_hook(self._hook_fn)
            self.handles.append(handle)
    
    def _hook_fn(self, module, input, output):
        """
        Hook function to capture intermediate attention computations.
        This is purely observational and doesn't modify the forward pass.
        
        Args:
            module: the attention module
            input: tuple of input tensors
            output: the attention output and attention weights
        """
        # The output from BERT's attention already contains attention probs
        attention_probs = output[1]  # This is the attention weights
        batch_size = attention_probs.size(0)
        
        # Get the value projections using existing module
        value = module.value(input[0])
        
        # Reshape for multi-head attention
        batch_size = value.size(0)
        
        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention output (this is what we want to capture)
        attention_output = torch.matmul(attention_probs, value)
        print (attention_probs.shape, value.shape, attention_output.shape)
        
        # Store both probabilities and value-weighted outputs
        self.attention_with_values.append({
            'attention_probs': attention_probs.detach(),
            'attention_values': attention_output.detach()
        })
        
    def forward(self, text):
        """
        Process text and return attention patterns with value products
        """
        self.attention_with_values = []  # Reset storage
        
        # Tokenize and run through model
        inputs = self.model.tokenizer(text, return_tensors="pt")
        outputs = self.model.model(**inputs, output_attentions=True)
        
        # Get tokens for reference
        tokens = self.model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return tokens, self.attention_with_values
    
    def visualize_value_weighted_attention(self, layer_idx, head_idx, tokens):
        """
        Visualize the attention outputs after value multiplication
        """
        # Get the attention values for specified layer
        attention_data = self.attention_with_values[layer_idx]
        
        # Get attention probs and value-weighted outputs for specified head
        attention_probs = attention_data['attention_probs'][0, head_idx].numpy()
        attention_values = attention_data['attention_values'][0, head_idx].numpy()

        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot original attention weights
        sns.heatmap(attention_probs, xticklabels=tokens, yticklabels=tokens,
                    ax=ax1, cmap='viridis')
        ax1.set_title(f'Layer {layer_idx} Head {head_idx}\nRaw Attention Weights')
        
         # Compute cosine similarity between output vectors        
        similarity_matrix = cosine_similarity(attention_values)
        sns.heatmap(similarity_matrix, xticklabels=tokens, yticklabels=tokens,
                    ax=ax2, cmap='viridis')
        ax2.set_title(f'Layer {layer_idx} Head {head_idx}\nValue-Weighted Attention')
        
        plt.tight_layout()
        return fig