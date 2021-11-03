import torch
import torch.nn as nn
from backbone_bert import BERTBackbone

class SentenceAgg(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(768, 1536)
		self.linear2 = nn.Linear(1536, 200)
	
	def __call__(self, x):
		'''
		inputs: (B, T, D) embeddings for the words
		obtain sentence embedding
		'''
		x = self.linear1(x)
		x = torch.mean(x, dim = -2)
		x = self.linear2(x)
		return x

class CondAug(nn.Module):
	def __init__(self):
		super().__init__()
		self.mean_weight = torch.normal(1, 0.5)
		self.mean_weight.requires_grad = True
		self.std_weight = torch.normal(1, 0.5)
		self.std_weight.requires_grad = True

	def __call__(self, x):
		return x # implement later

class Image2TextAttention(nn.Module):
	def __init__(self, output_shape, num_head=1):
		super().__init__()
		assert(output_shape%num_head == 0)
		self.attentor = nn.MultiheadAttention(output_shape, num_head, batch_first=True)

	def __call__(self, img, txt):
		'''
		img = (N, Di, H, W)
		txt = (N, T, Dt)
		'''
		img = torch.reshape(img, (img.shape[0], img.shape[1], -1))
		img = torch.transpose(img, 1, 2)
		attn_output, attn_output_weights = self.attentor(img, txt, txt)
		# = (N, H*W, D)
		attn_output_weights = torch.transpose(attn_output_weights, 1, 2)
		attn_output_weights = torch.reshape(attn_output_weights, 
			(attn_output_weights.shape[0], attn_output_weights.shape[1], -1))
		return attn_output_weights






