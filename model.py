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

    def print_model_info(self):
        """Print detailed model architecture with parameter counts"""
        print(f"=== TinyBERT Architecture Overview ===\n")
        
        # Model dimensions
        print("Global Parameters:")
        print(f"Hidden Size: {self.model.config.hidden_size}")
        print(f"Number of Attention Heads: {self.num_heads}")
        print(f"Head Dimension: {self.model.config.hidden_size // self.num_heads}")
        print(f"Number of Layers: {self.num_layers}")
        print(f"Intermediate Size: {self.model.config.intermediate_size}")
        print(f"Max Position Embeddings: {self.model.config.max_position_embeddings}\n")
        
        # Embeddings
        print("Embedding Layer:")
        print(f"└── Word Embeddings: {self.model.embeddings.word_embeddings.weight.shape}")
        print(f"└── Position Embeddings: {self.model.embeddings.position_embeddings.weight.shape}")
        print(f"└── Token Type Embeddings: {self.model.embeddings.token_type_embeddings.weight.shape}")
        print(f"└── Layer Norm: {tuple(self.model.embeddings.LayerNorm.weight.shape)}\n")
        
        # Transformer Layers
        print("Transformer Layers:")
        for layer_idx in range(self.num_layers):
            layer = self.model.encoder.layer[layer_idx]
            print(f"Layer {layer_idx}:")
            
            # Attention
            print("└── Self-Attention:")
            print(f"    ├── Query: {tuple(layer.attention.self.query.weight.shape)}")
            print(f"    ├── Key: {tuple(layer.attention.self.key.weight.shape)}")
            print(f"    ├── Value: {tuple(layer.attention.self.value.weight.shape)}")
            print(f"    └── Output: {tuple(layer.attention.output.dense.weight.shape)}")
            
            # FFN
            print("└── Feed Forward Network:")
            print(f"    ├── Intermediate: {tuple(layer.intermediate.dense.weight.shape)}")
            print(f"    └── Output: {tuple(layer.output.dense.weight.shape)}\n")
            
        # Total Parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params:,}")
        
    # def get_attention_maps(self, text):
    #     """Get attention maps for input text."""
    #     inputs = self.tokenizer(text, return_tensors='pt')
    #     with torch.no_grad():
    #         outputs = self.model(**inputs, output_attentions=True)
        
    #     tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    #     return outputs.attentions, tokens
    
    # def visualize_head(self, text, layer_num=0, head_num=0):
    #     """Visualize a single attention head."""
    #     attention_maps, tokens = self.get_attention_maps(text)
    #     attention = attention_maps[layer_num][0, head_num].numpy()
        
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(attention, 
    #                 xticklabels=tokens,
    #                 yticklabels=tokens,
    #                 cmap='viridis',
    #                 square=True)
        
    #     plt.title(f'TinyBERT Attention Pattern\nLayer {layer_num}, Head {head_num}')
    #     plt.xlabel('Key Tokens')
    #     plt.ylabel('Query Tokens')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.tight_layout()
    #     return plt

    # def analyze_layer_heads(self, text, layer_num=0):
    #     """Analyze all heads in a given layer."""
    #     attention_maps, tokens = self.get_attention_maps(text)
        
    #     # Calculate grid dimensions based on number of heads
    #     grid_size = int(np.ceil(np.sqrt(self.num_heads)))
    #     fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    #     axes = axes.ravel()
        
    #     for head in range(self.num_heads):
    #         attention = attention_maps[layer_num][0, head].numpy()
            
    #         sns.heatmap(attention,
    #                     xticklabels=tokens,
    #                     yticklabels=tokens,
    #                     cmap='viridis',
    #                     square=True,
    #                     ax=axes[head])
            
    #         axes[head].set_title(f'Head {head}')
    #         axes[head].set_xticklabels(tokens, rotation=45, ha='right')
    #         axes[head].set_yticklabels(tokens, rotation=0)
        
    #     # Hide empty subplots if any
    #     for idx in range(self.num_heads, len(axes)):
    #         axes[idx].axis('off')
        
    #     plt.suptitle(f'All Attention Heads in Layer {layer_num}', fontsize=16)
    #     plt.tight_layout()
    #     return plt

    # def compare_attention_patterns(self, texts, layer_num=0, head_num=0):
    #     """Compare attention patterns across different inputs."""
    #     fig, axes = plt.subplots(1, len(texts), figsize=(20, 6))
    #     if len(texts) == 1:
    #         axes = [axes]
        
    #     for idx, text in enumerate(texts):
    #         attention_maps, tokens = self.get_attention_maps(text)
    #         attention = attention_maps[layer_num][0, head_num].numpy()
            
    #         sns.heatmap(attention,
    #                     xticklabels=tokens,
    #                     yticklabels=tokens,
    #                     cmap='viridis',
    #                     square=True,
    #                     ax=axes[idx])
            
    #         axes[idx].set_title(f'Text {idx + 1}')
    #         axes[idx].set_xticklabels(tokens, rotation=45, ha='right')
        
    #     plt.suptitle(f'Attention Patterns Comparison\nLayer {layer_num}, Head {head_num}', fontsize=16)
    #     plt.tight_layout()
    #     return plt