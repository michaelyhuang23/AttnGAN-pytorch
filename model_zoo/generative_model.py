import torch
from torch import nn
from transformers import AutoTokenizer
from backbone_bert import BERTBackbone
from auxiliary_layers import SentenceAgg, CondAug, Image2TextAttention, ImgGen
from generative_layers import FirstGen, SecondGen, ThirdGen

class AttnGAN(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone  = BERTBackbone()
		self.sentence_alg = SentenceAgg()
		self.cond_aug = CondAug()
		self.first_part = FirstGen()
		self.imggen1 = ImgGen(64)
		self.attn_1 = Image2TextAttention(64)
		self.second_part = SecondGen()
		self.imggen2 = ImgGen(64)
		self.attn_2 = Image2TextAttention(64)
		self.third_part = ThirdGen()
		self.imggen3 = ImgGen(64)

	def forward(self, x):
		print(x)
		word_fs = self.backbone(x)
		sentence_f = self.sentence_alg(word_fs)
		sentence_f = self.cond_aug(sentence_f)
		first_stage = self.first_part(sentence_f)
		first_img = self.imggen1(first_stage)
		output = self.attn_1(first_stage, word_fs)
		print(output.shape, first_stage.shape)
		second_stage = output + first_stage
		# add weighing constant
		second_stage = self.second_part(second_stage)
		second_img = self.imggen2(second_stage)
		third_stage = self.attn_1(second_stage, word_fs) + second_stage
		third_stage = self.third_part(third_stage)
		third_img = self.imggen3(third_stage)
		return first_img, second_img, third_img


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world!", return_tensors="pt")
model = AttnGAN()
model.eval()
img1, img2, img3 = model(inputs)

print(img1.shape, img2.shape, img3.shape)