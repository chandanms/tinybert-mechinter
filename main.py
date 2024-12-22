from model import TinyBertModel
from visualize import AttentionVisualizer
import torch

tiny_bert = TinyBertModel()

visualizer = AttentionVisualizer(tiny_bert)

visualizer.visualize_layer("The cat sat on the mat.", 0)