from torch import nn
import torch
from auxiliary_layers import ResBlock

class discrim1(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, (5,5), padding=0) # 60 by 60
		self.relu = nn.ReLU()
		self.res1 = ResBlock()
		self.maxpool1 = nn.MaxPool2d((2,2),2) # 30 by 30
		self.res2 = ResBlock()
		self.maxpool2 = nn.MaxPool2d((2,2),2) # 15 by 15
		self.res3 = nn.ResBlock()
		self.maxpool3 = nn.MaxPool2d((3,3),2) # 7 by 7
		self.res4 = nn.ResBlock() # then perform global average pooling
		self.final = nn.Linear(64, 200)
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	def forward(self, x, sentence_f):
		'''
		take in 64 by 64
		output 200 1D data
		'''
		x = self.relu(self.conv1(x))
		x = self.res1(x) + x
		x = self.maxpool1(x)
		x = self.res2(x) + x
		x = self.maxpool2(x)
		x = self.res3(x) + x
		x = self.maxpool3(x)
		x = self.res4(x) + x
		x = torch.mean(x, (-2,-1))
		x = self.final(x) # try confusion technique?
		return self.cos(x, sentence_f)

class discrim2(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, (7,7), padding=0) # 122 by 122
		self.relu = nn.ReLU()
		self.res1 = ResBlock()
		self.maxpool1 = nn.MaxPool2d((2,2),2) # 61 by 61
		self.res2 = ResBlock()
		self.maxpool2 = nn.MaxPool2d((3,3),2) # 30 by 30
		self.res3 = ResBlock()
		self.maxpool3 = nn.MaxPool2d((2,2),2) # 15 by 15
		self.res4 = nn.ResBlock()
		self.maxpool4 = nn.MaxPool2d((3,3),2) # 7 by 7
		self.res5 = nn.ResBlock() # then perform global average pooling
		self.final = nn.Linear(64, 200)
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	def forward(self, x, sentence_f):
		'''
		take in 128 by 128
		output 200 1D data
		'''
		x = self.relu(self.conv1(x))
		x = self.res1(x) + x
		x = self.maxpool1(x)
		x = self.res2(x) + x
		x = self.maxpool2(x)
		x = self.res3(x) + x
		x = self.maxpool3(x)
		x = self.res4(x) + x
		x = self.maxpool4(x)
		x = self.res5(x) + x
		x = torch.mean(x, (-2,-1))
		x = self.final(x) # try confusion technique?
		return self.cos(sentence_f, x)

class discrim3(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, (7,7), stride=2, padding=0) # 125 by 125
		self.relu = nn.ReLU()
		self.res1 = ResBlock()
		self.conv2 = nn.Conv2d(64, 64, (6,6), padding=0) # 120 by 120
		self.res2 = ResBlock()
		self.maxpool1 = nn.MaxPool2d((2,2),2) # 60 by 60
		self.res3 = ResBlock()
		self.maxpool2 = nn.MaxPool2d((2,2),2) # 30 by 30
		self.res4 = nn.ResBlock()
		self.maxpool3 = nn.MaxPool2d((2,2),2) # 15 by 15
		self.res5 = nn.ResBlock() # then perform global average pooling
		self.maxpool4 = nn.MaxPool2d((3,3),2) # 7 by 7
		self.res6 = nn.ResBlock()
		self.final = nn.Linear(64, 200)
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	def forward(self, x, sentence_f):
		'''
		take in 256 by 256
		sentence_f 
		output 200 1D data
		'''
		x = self.relu(self.conv1(x))
		x = self.res1(x) + x
		x = self.conv2(x)
		x = self.res2(x) + x
		x = self.maxpool1(x)
		x = self.res3(x) + x
		x = self.maxpool2(x)
		x = self.res4(x) + x
		x = self.maxpool3(x)
		x = self.res5(x) + x
		x = self.maxpool4(x)
		x = self.res4(x) + x
		x = torch.mean(x, (-2,-1))
		x = self.final(x) # try confusion technique?
		return self.cos(sentence_f, x)



