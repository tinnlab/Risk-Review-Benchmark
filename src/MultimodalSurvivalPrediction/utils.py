"""Utils"""

import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


# The dictionary of the input data path


def setup_seed(seed):
	"""
	Set random seed for torch.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True

def test_gpu():
	"""
	Detect the hardware: GPU or CPU?
	"""
	print('GPU？', torch.cuda.is_available())
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('The device is ', device)
	return device

def evaluate_model(c_index_arr):
	"""
	Calculate the Mean and the Standard Deviation about c-index array.
	"""
	m = np.sum(c_index_arr, axis=0) / len(c_index_arr)
	s = np.std(c_index_arr)
	return m, s


def get_dataloaders(mydataset_train, mydataset_val, batch_size):
	dataloaders = {}
	dataloaders['train'] = DataLoader(mydataset_train, batch_size=batch_size, drop_last=True)
	dataloaders['val'] = DataLoader(mydataset_val, batch_size=batch_size)
	return dataloaders


def get_dataloaders_test(mydataset_test, batch_size):
	dataloaders = {}
	dataloaders['test'] = DataLoader(mydataset_test, batch_size=batch_size)
	return dataloaders


def compose_run_tag(model, lr, dataloaders, log_dir, suffix=''):
	"""
	Make the tag about modality and learning rate.
	"""
	def add_string(string, addition, sep='_'):
		if not string:
			return addition
		else: return string + sep + addition

	data = None
	for modality in model.data_modalities:
		data = add_string(data, modality)

	run_tag = f'{data}_lr{lr}'

	run_tag += suffix

	print(f'Run tag: "{run_tag}"')

	tb_log_dir = os.path.join(log_dir, run_tag)

	return run_tag


def save_5fold_results(c_index_arr, run_tag):
	"""
	Save the results after 5 fold cross validation.
	"""
	m, s = evaluate_model(c_index_arr)
	with open(f'proposed_{run_tag}.txt', 'w') as file:
		file.write(str(c_index_arr))
		file.write(f"\n Mean: {m}")
		file.write(f"\n Std: {s}")
	file.close()

























