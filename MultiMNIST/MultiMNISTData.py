
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
import random
from torchvision.utils import save_image

import math

import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import pickle

import numpy as np
from torch.utils import data
import h5py

from os import listdir
from os.path import isfile, join

class DatasetMultiMNIST(Dataset):
	def __init__(self, DataDir,Noise=None):
		self.DataDir = DataDir
		self.n_classes = 10
		self.Noise=Noise
	def __len__(self):

		ImageFiles = [f for f in listdir(self.DataDir) if isfile(join(self.DataDir, f))]
	
		return len(ImageFiles)

	def __getitem__(self, index):
		


		#if self.Noise==None:
		transform = torchvision.transforms.Compose(
		[torchvision.transforms.Resize((128, 128)), torchvision.transforms.ToTensor()])
		# elif self.Noise=="Gauss":
		# 	transform = torchvision.transforms.Compose(
		# 	[torchvision.transforms.Resize((128,128)),
		# 	 torchvision.transforms.ToTensor(),GaussianNoise(),
		# 	 torchvision.transforms.ToTensor()])


		ImageFiles = [f for f in listdir(self.DataDir) if isfile(join(self.DataDir, f))]
		
		img = Image.open(self.DataDir+"/"+ImageFiles[index])
		img=transform(img)
		
	
		label = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
		
		y=ImageFiles[index].split("_")[1].split(".")[0]
		
		for i in y:
			label[int(i)]=1.0	
		# label_index = np.where(self.labels == label)[0][0]

		x=img
		

		#convert to RGB
		x = x.repeat(1,3,1,1)
		for j in range(x.shape[0]):
			x[j,0:1] *= 1
			x[j,1:2] *=0.5
			x[j,2:3] *= 0.5

		
		img=x.squeeze(0)
	
		if self.Noise=="Gauss":
			AddNoise=GaussianNoise()
			img=AddNoise(img)
		return img, label  # returns features (image) and target index in self.labels, which corresponds to the target softmax index in the model


class TORGB(object):

	 ####Original MNIST to RGB
	def __call__(self, sample):
	
		x=sample 

		x = x.repeat(1,3,1,1)
		for j in range(x.shape[0]):
			x[j,0:1] *= 0.5
			x[j,1:2] *=1.0
			x[j,2:3] *=1.0

		
		img=x.squeeze(0)

		return img




class GaussianNoise(object):

	 ####Original MNIST to RGB
	def __call__(self, sample):

		x=sample 
		image=x.detach().cpu().numpy() 
		row,col,ch= image.shape
		mean = 0
		var = 0.6
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		noisy = image + gauss
		x=torch.tensor(noisy,dtype=torch.float32)
		return x






if __name__ == "__main__":
	print("this codes load MNIST data")


	import os
	#abspath = os.path.abspath(__file__)
	#dname = os.path.dirname(abspath)
	#os.chdir(dname)
	#os.chdir("DataAndExtractor/")
	#print("working dir: "+str(dname))


	#dataset_train, dataset_valid,dataset_test=Get_MNISTdata(Data="MNISTChinese",Noise="Gauss")
	#print("dataset")
	#print(len(dataset_train))
	#img,label=dataset_train.__getitem__(10000)
	#print(img.shape)
	#print(label)
	
	DataSet=DatasetMultiMNIST("../../data/double_mnist/test/",Noise="Gauss")
	print("Data length")
	print(DataSet.__len__())


	img,label=DataSet.__getitem__(1)
	print(img.shape)
	print(label)

	save_image(img, 'testing1.png')


	# save_image(torch.tensor(obs).squeeze(0), 'testing6.png')

