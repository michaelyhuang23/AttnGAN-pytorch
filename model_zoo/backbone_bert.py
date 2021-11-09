import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BERTBackbone(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = AutoModel.from_pretrained("bert-base-cased")

	def __call__(self, inputs):
		'''
		input dict
		'attention_mask': (B, T)
		'input_ids': (B, T)
		'labels': (B)
		outputs
		(B, T, D) where D is the last layer dim
		'''
		return self.model(**inputs).last_hidden_state