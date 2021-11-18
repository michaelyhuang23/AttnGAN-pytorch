import torch
import torch.nn as nn
from backbone_bert import BERTBackbone

class SentenceAgg(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(768, 1536)
		self.relu = nn.ReLU()
		self.linear2 = nn.Linear(1536, 200)
	
	def forward(self, x):
		'''
		inputs: (B, T, D) embeddings for the words
		obtain sentence embedding
		'''
		x = self.relu(self.linear1(x))
		x = torch.mean(x, dim = -2)
		x = self.relu(self.linear2(x))
		return x

class CondAug(nn.Module):
	def __init__(self):
		super().__init__()
		self.mean_weight = torch.normal(torch.tensor(1.0), torch.tensor(0.5))
		self.mean_weight.requires_grad = True
		self.std_weight = torch.normal(torch.tensor(1.0), torch.tensor(0.5))
		self.std_weight.requires_grad = True

	def forward(self, x):
		return x # implement later

class Image2TextAttention(nn.Module):
	def __init__(self, output_shape, num_head=1):
		super().__init__()
		assert(output_shape%num_head == 0)
		self.texthead = nn.Linear(768, 64)
		self.attentor = nn.MultiheadAttention(output_shape, num_head, batch_first=True)

	def forward(self, img, txt):
		'''
		img = (N, Di, H, W)
		txt = (N, T, Dt)
		'''
		height = img.shape[2]
		width = img.shape[3]
		img = torch.reshape(img, (img.shape[0], img.shape[1], -1))
		img = torch.transpose(img, 1, 2)
		txt = self.texthead(txt)
		attn_output, attn_output_weights = self.attentor(img, txt, txt)
		# = (N, H*W, D)
		attn_output = torch.transpose(attn_output, 1, 2)
		attn_output = torch.reshape(attn_output, 
			(attn_output.shape[0], attn_output.shape[1], height, width))
		return attn_output

class ResBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(64, 128, (3,3), 1, padding=1)
		self.batchnorm1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
		self.relu = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 64, (3,3), 1, padding=1)
		self.batchnorm2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
	def forward(self, inputs):
		x = self.relu(self.conv1(inputs))
		x = self.batchnorm1(x)
		x = self.relu(self.conv2(x))
		x = self.batchnorm2(x+inputs)
		return x

class ImgGen(nn.Module):
	def __init__(self, input_shape):
		super().__init__()
		self.conv1 = nn.Conv2d(input_shape, 3, (3,3), padding=1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		return self.sigmoid(self.conv1(x))

