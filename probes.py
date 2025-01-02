import torch
from torch import nn
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

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
    

class CounterfactualAttentionProbe(AttentionProbe):
    def __init__(self, bert_model):
        super().__init__(bert_model)
        
    def compare_statements(self, original_text, counterfactual_text):
        """
        Compare attention patterns between original and counterfactual statements
        
        Args:
            original_text (str): The original statement
            counterfactual_text (str): The counterfactual version
        
        Returns:
            tuple: Original and counterfactual tokens and attention data
        """
        # Process both statements
        orig_tokens, orig_attention = self.forward(original_text)
        cf_tokens, cf_attention = self.forward(counterfactual_text)
        
        return {
            'original': {'tokens': orig_tokens, 'attention': orig_attention},
            'counterfactual': {'tokens': cf_tokens, 'attention': cf_attention}
        }
    
    def visualize_attention_diff(self, layer_idx, head_idx, orig_data, cf_data):
        """
        Visualize the difference in attention patterns between original and counterfactual
        
        Args:
            layer_idx (int): Index of the layer to analyze
            head_idx (int): Index of the attention head to analyze
            orig_data (dict): Original statement data
            cf_data (dict): Counterfactual statement data
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Original attention patterns
        orig_attn = orig_data['attention'][layer_idx]['attention_probs'][0, head_idx].numpy()
        orig_values = orig_data['attention'][layer_idx]['attention_values'][0, head_idx].numpy()
        orig_tokens = orig_data['tokens']
        
        # Counterfactual attention patterns
        cf_attn = cf_data['attention'][layer_idx]['attention_probs'][0, head_idx].numpy()
        cf_values = cf_data['attention'][layer_idx]['attention_values'][0, head_idx].numpy()
        cf_tokens = cf_data['tokens']
        
        # Plot original attention
        sns.heatmap(orig_attn, xticklabels=orig_tokens, yticklabels=orig_tokens,
                   ax=axes[0,0], cmap='viridis')
        axes[0,0].set_title(f'Original Statement\nAttention Weights')
        
        # Plot counterfactual attention
        sns.heatmap(cf_attn, xticklabels=cf_tokens, yticklabels=cf_tokens,
                   ax=axes[0,1], cmap='viridis')
        axes[0,1].set_title(f'Counterfactual Statement\nAttention Weights')
        
        # Plot original value-weighted attention
        orig_similarity = cosine_similarity(orig_values)
        sns.heatmap(orig_similarity, xticklabels=orig_tokens, yticklabels=orig_tokens,
                   ax=axes[1,0], cmap='viridis')
        axes[1,0].set_title(f'Original Statement\nValue-Weighted Attention')
        
        # Plot counterfactual value-weighted attention
        cf_similarity = cosine_similarity(cf_values)
        sns.heatmap(cf_similarity, xticklabels=cf_tokens, yticklabels=cf_tokens,
                   ax=axes[1,1], cmap='viridis')
        axes[1,1].set_title(f'Counterfactual Statement\nValue-Weighted Attention')
        
        plt.suptitle(f'Layer {layer_idx} Head {head_idx}\nAttention Pattern Comparison', y=1.02)
        plt.tight_layout()
        return fig
    
    def compute_attention_differences(self, layer_idx, head_idx, orig_data, cf_data):
        """
        Compute statistical differences between original and counterfactual attention patterns
        
        Args:
            layer_idx (int): Index of the layer to analyze
            head_idx (int): Index of the attention head to analyze
            orig_data (dict): Original statement data
            cf_data (dict): Counterfactual statement data
            
        Returns:
            dict: Dictionary containing various difference metrics
        """
        # Get attention patterns
        orig_attn = orig_data['attention'][layer_idx]['attention_probs'][0, head_idx]
        cf_attn = cf_data['attention'][layer_idx]['attention_probs'][0, head_idx]
        
        # Get value-weighted patterns
        orig_values = orig_data['attention'][layer_idx]['attention_values'][0, head_idx]
        cf_values = cf_data['attention'][layer_idx]['attention_values'][0, head_idx]

        ## TO DO: Figure out a way to compare attention differences for different sentence sizes
        
        # Compute differences
        attn_diff = {
            'max_attention_diff': torch.max(torch.abs(orig_attn - cf_attn)).item(),
            'mean_attention_diff': torch.mean(torch.abs(orig_attn - cf_attn)).item(),
            'attention_pattern_correlation': torch.corrcoef(
                torch.stack([orig_attn.flatten(), cf_attn.flatten()])
            )[0,1].item(),
            'value_output_correlation': torch.corrcoef(
                torch.stack([orig_values.flatten(), cf_values.flatten()])
            )[0,1].item()
        }
        
        return attn_diff


class NegationProbe(AttentionProbe):
    def __init__(self, bert_model):
        super().__init__(bert_model)
        self.negation_tokens = {'not', 'never', "n't", 'no', 'neither', 'nor'}
        
    def analyze_negation_pairs(self, sentence_pairs: List[Tuple[str, str]]):
        """
        Analyze multiple pairs of sentences with and without negation.
        
        Args:
            sentence_pairs: List of (original, negated) sentence pairs
            
        Returns:
            Dict containing analysis results for each layer and head
        """
        results = {}
        max_len = max(len(self.forward(neg)[0]) for _, neg in sentence_pairs)
        for layer_idx in range(self.num_layers):
            results[layer_idx] = {}
            for head_idx in range(self.num_heads):
                head_from_neg = []
                head_to_neg = []                
                
                for _, neg in sentence_pairs:
                    neg_tokens, neg_attention = self.forward(neg)
                    
                    # Track negation token attention
                    neg_attention, neg_influence = self._compute_negation_attention(
                        layer_idx, head_idx,
                        neg_tokens,
                        neg_attention
                    )

                    if isinstance(neg_attention, torch.Tensor):
                        neg_attention = neg_attention.cpu().numpy()
                        neg_influence = neg_influence.cpu().numpy()
                    
                    padded_attn = np.zeros(max_len)
                    padded_infl = np.zeros(max_len)

                    padded_attn[:len(neg_attention)] = neg_attention
                    padded_infl[:len(neg_influence)] = neg_influence

                    neg_attention = padded_attn
                    neg_influence = padded_infl

                    head_from_neg.append(neg_attention)
                    head_to_neg.append(neg_influence)
                
                # Aggregate metrics across all pairs
                results[layer_idx][head_idx] = {
                    'from_neg_to_tokens': np.mean(head_from_neg, axis=0),
                    'to_neg_from_tokens': np.mean(head_to_neg, axis=0)
                }                
        
        return results
    
    def _compute_negation_attention(self, layer_idx: int, head_idx: int, 
                                  tokens: List[str], attention_data: List[Dict]) -> Dict:
        """
        Compute attention patterns specifically related to negation tokens.
        """
        # Find positions of negation tokens
        neg_positions = [i for i, t in enumerate(tokens) 
                        if any(neg in t.lower() for neg in self.negation_tokens)]
        if not neg_positions:
            # Return zero tensors if no negation found
            zero_attn = torch.zeros(len(tokens))
            return zero_attn, zero_attn
        
        # Get attention weights for this layer/head
        attn_weights = attention_data[layer_idx]['attention_probs'][0, head_idx]
        
        ## TO DO: Investigate if taking average attention hides any important details
        # # Average attention from negation tokens to all other tokens
        # neg_attention = torch.mean(attn_weights[neg_positions, :]).item()
        
        # # Average attention to negation tokens from all other tokens
        # neg_influence = torch.mean(attn_weights[:, neg_positions]).item()

        # Attention from negation token to all other tokens
        neg_attention = attn_weights[neg_positions, :][0]

        neg_influence = attn_weights[:, neg_positions].reshape(-1)

        return neg_attention, neg_influence
    
    
    def visualize_negation_head_rankings(self, results: Dict):
        """
        Visualize which heads appear most responsive to negation.
        """
        # Create matrices for different metrics
        metrics = ['neg_token_attention', 'neg_token_influence', 
                  'attention_pattern_correlation']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        
        for idx, metric in enumerate(metrics):
            data = np.zeros((self.num_layers, self.num_heads))
            
            for layer in range(self.num_layers):
                for head in range(self.num_heads):
                    data[layer, head] = results[layer][head][metric]
            
            sns.heatmap(data, ax=axes[idx], cmap='viridis',
                       xticklabels=[f'Head {i}' for i in range(self.num_heads)],
                       yticklabels=[f'Layer {i}' for i in range(self.num_layers)])
            axes[idx].set_title(f'{metric.replace("_", " ").title()}')
        
        plt.tight_layout()
        return fig