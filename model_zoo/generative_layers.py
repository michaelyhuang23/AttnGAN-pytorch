import torch
from torch import nn

class FirstGen(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(200, 18432)
		self.relu = nn.ReLU()
		self.convup1 = nn.ConvTranspose2d(512, 256, (4,4), stride=1, padding=0)
		self.convup2 = nn.ConvTranspose2d(256, 128, (4,4), stride=2, padding=0)
		self.convup3 = nn.ConvTranspose2d(128, 64, (7,7), stride=3, padding=0)

	def forward(self, x):
		'''
		x is of shape (B, D)
		output is of shape (B, 64, 64, 64)
		'''
		x = self.relu(self.linear1(x))
		x = torch.reshape(x.shape[0], 512, 6, 6)
		x = self.relu(self.convup1(x))
		x = self.relu(self.convup2(x))
		x = self.relu(self.convup3(x)) # add batch_norms
		return x

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

class SecondGen(nn.Module):
	def __init__(self):
		super().__init__()
		self.res1 = ResBlock()
		self.res2 = ResBlock()
		self.relu = nn.ReLU()
		self.convup1 = nn.ConvTranspose2d(64, 64, (4,4), stride=2, padding=1)

	def forward(self, x):
		'''
		x is of shape (B, D)
		output is of shape (B, 64, 128, 128)
		'''
		x = self.res1(x)
		x = self.res2(x)
		x = self.relu(self.convup1(x))
		return x

class ThirdGen(nn.Module):
	def __init__(self):
		super().__init__()
		self.res1 = ResBlock()
		self.res2 = ResBlock()
		self.relu = nn.ReLU()
		self.convup1 = nn.ConvTranspose2d(64, 64, (4,4), stride=2, padding=1)

	def forward(self, x):
		'''
		x is of shape (B, D)
		output is of shape (B, 64, 256, 256)
		'''
		x = self.res1(x)
		x = self.res2(x)
		x = self.relu(self.convup1(x))
		return x
