import torch
from torch import nn

from backbone_bert import BERTBackbone
from auxiliary_layers import SentenceAgg, CondAug, Image2TextAttention, ImgGen
from generative_layers import FirstGen, SecondGen, ThirdGen

class AttnGAN(nn.Module):
	def __init__(self):
		self.backbone  = BERTBackbone()
		self.sentence_alg = SentenceAgg()
		self.cond_aug = CondAug()
		self.first_part = FirstGen()
		self.attn_1 = Image2TextAttention(64)
		self.second_part = SecondGen()
		self.attn_2 = Image2TextAttention(64)
		self.third_part = ThirdGen()
		pass

	def foward(self, x):
		word_fs = self.backbone(x)
		sentence_f = self.sentence_alg(word_fs)
		sentence_f = self.cond_aug(sentence_f)
		first_stage = self.first_part(sentence_f)
		first_img = ImgGen(first_stage)
		second_stage = self.attn_1(first_stage, word_fs) + first_stage
		# add weighing constant
		second_stage = self.second_part(second_stage)
		second_img = ImgGen(second_stage)
		third_stage = self.attn_1(second_stage, word_fs) + second_stage
		third_stage = self.third_part(third_stage)
		third_img = ImgGen(third_stage)
		return first_img, second_img, third_img
		
