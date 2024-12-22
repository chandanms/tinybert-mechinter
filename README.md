# Mechanistic interpretability of TinyBERT

This project provides tools for analyzing BERT attention mechanisms using counterfactual statements. It enables detailed investigation of how attention patterns and value transformations change when semantically related but meaningfully different statements are processed.

Please note that this project is not complete and only my scratchbook to understand the mechanisms better

## Features

- Attention pattern visualization before and after value matrix transformation
- Counterfactual analysis comparing attention patterns between related statements
- Statistical metrics for quantifying attention pattern differences
- Support for both raw attention weights and value-weighted attention analysis
- Layer-wise and head-wise analysis capabilities


### Requirements

- PyTorch
- Transformers
- Seaborn
- Matplotlib
- NumPy
- TinyBERT (prajjwal1/bert-tiny)

## Usage

### Basic Model Setup

```python
from model import TinyBertModel
from probing import AttentionProbe, CounterfactualAttentionProbe

# Initialize the model
model = TinyBertModel()

# Print model architecture
model.print_model_info()
```

### Analyzing Attention Patterns

```python
# Define statement pairs
statements = (
    "The cat chased the mouse",
    "The mouse chased the cat"
)

# Analyze specific layer and head
layer_idx = 0
head_idx = 0

# Run analysis
fig, differences = analyze_counterfactual_pair(model, statements, layer_idx, head_idx)
```

## Understanding the Output

### Visualization Components

The visualization includes four heatmaps:
1. Original statement attention weights
2. Counterfactual statement attention weights
3. Original value-weighted attention
4. Counterfactual value-weighted attention

### Statistical Metrics

- `max_attention_diff`: Maximum absolute difference in attention weights
- `mean_attention_diff`: Average absolute difference in attention weights
- `attention_pattern_correlation`: Correlation between original and counterfactual attention patterns
- `value_output_correlation`: Correlation between original and counterfactual value-weighted outputs

### Interpreting Results

#### Significant Patterns to Look For:

1. **Raw Attention Weight Changes**
   - Strong attention bands shifting positions
   - Changes in diagonal patterns (self-attention)
   - Shifts at syntactically important positions

2. **Value-Weighted Attention Changes**
   - Changes in semantic clustering
   - Alterations in relationship strengths between key words
   - Preservation or disruption of semantic relationships

3. **Statistical Indicators**
   - High max_attention_diff (>0.3) indicates significant local changes
   - Low attention_pattern_correlation (<0.7) suggests substantial pattern reorganization
   - Discrepancies between value_output_correlation and attention_pattern_correlation indicate different semantic processing

## Architecture

### TinyBertModel
Base class for loading and managing the BERT model:
- Handles model initialization
- Provides inference capabilities
- Exposes model architecture information

### AttentionProbe
Core probing functionality:
- Captures intermediate attention computations
- Implements visualization for attention patterns
- Handles value matrix transformations

### CounterfactualAttentionProbe
Extends AttentionProbe with counterfactual analysis:
- Compares attention patterns between statement pairs
- Provides statistical metrics for differences
- Visualizes comparative attention patterns

## References

- https://www.neelnanda.io/mechanistic-interpretability/othello
- https://github.com/karpathy/nanoGPT
