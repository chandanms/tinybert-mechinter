import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionVisualizer:
    def __init__(self, model):
        """
        Initialize the attention visualizer with a TinyBERT model.
        
        Args:
            model: TinyBertModel instance
        """
        self.model = model
        self.num_layers = model.num_layers
        self.num_heads = model.num_heads

        print (self.num_layers, self.num_heads)
        
    def get_attention_maps(self, text):
        """
        Get attention maps for all layers and heads.
        
        Args:
            text: Input text string
            
        Returns:
            tokens: List of tokens
            attention_maps: List of attention matrices for each layer
        """
        # Get model outputs with attention
        inputs = self.model.tokenizer(text, return_tensors="pt")
        outputs = self.model.model(**inputs, output_attentions=True)
        
        # Get tokens for visualization
        tokens = self.model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get attention matrices (shape: [layers, heads, seq_len, seq_len])
        attention_maps = [layer_attention[0].detach().numpy() 
                         for layer_attention in outputs.attentions]
        
        return tokens, attention_maps
    
    def plot_attention_head(self, attention_matrix, tokens, ax=None, title=None):
        """
        Plot a single attention head's attention pattern.
        
        Args:
            attention_matrix: 2D numpy array of attention weights
            tokens: List of tokens
            ax: Matplotlib axis object
            title: Title for the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
            
        # Create heatmap
        sns.heatmap(attention_matrix, 
                    xticklabels=tokens,
                    yticklabels=tokens,
                    ax=ax,
                    cmap='viridis',
                    cbar=True)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens, rotation=0)
        
        if title:
            ax.set_title(title)
            
    def visualize_layer(self, text, layer_idx):
        """
        Visualize all attention heads in a specific layer.
        
        Args:
            text: Input text string
            layer_idx: Index of the layer to visualize
        """
        tokens, attention_maps = self.get_attention_maps(text)
        
        # Create subplot grid for all heads in the layer
        fig, axes = plt.subplots(2, self.num_heads//2, 
                                figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot each attention head
        layer_attention = attention_maps[layer_idx]
        for head_idx in range(self.num_heads):
            head_attention = layer_attention[head_idx]
            title = f'Layer {layer_idx}, Head {head_idx}'
            self.plot_attention_head(head_attention, tokens, 
                                   axes[head_idx], title)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_aggregated_attention(self, text, layer_idx):
        """
        Visualize the aggregated attention pattern across all heads in a layer.
        
        Args:
            text: Input text string
            layer_idx: Index of the layer to visualize
        """
        tokens, attention_maps = self.get_attention_maps(text)
        
        # Average attention across all heads
        layer_attention = attention_maps[layer_idx]
        aggregated_attention = np.mean(layer_attention, axis=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 12))
        self.plot_attention_head(aggregated_attention, tokens, ax,
                               f'Layer {layer_idx} - Aggregated Attention')
        
        plt.tight_layout()
        plt.show()