"""MyDataset"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def preprocess_clinical_data(dataframes, cate_cols):
	target_data = dataframes['survival']
	clin_data = dataframes['clinical']
	nume_cols = list(set(clin_data.columns) - set(cate_cols))
	clin_data_categorical = clin_data[cate_cols]
	clin_data_continuous = clin_data[nume_cols]

	return clin_data_categorical, clin_data_continuous, target_data


# def preprocess_clinical_data_test(dataframes, cate_cols):
# 	clin_data = dataframes['clinical']
# 	nume_cols = list(set(clin_data.columns) - set(cate_cols))
# 	clin_data_categorical = clin_data[cate_cols]
# 	clin_data_continuous = clin_data[nume_cols]
#
# 	return clin_data_categorical, clin_data_continuous


class MyDataset(torch.utils.data.Dataset):
	def __init__(self, modalities, dataframes, cate_cols):
		"""
		Parameters
		----------
		modalities: list
			Used modalities

		data_path: dict
			The path of used data.

		Returns
		-------
		data: dictionary
			{'clin_data_categorical': ,..,'mRNA': ...}

		data_label: dictionary
			{'label':[[event, time]]}
		"""
		super(MyDataset, self).__init__()
		self.data_modalities = modalities
		# label
		clin_data_categorical, clin_data_continuous, target_data = preprocess_clinical_data(dataframes, cate_cols)
		self.target = target_data.values.tolist()

		# clinical
		if 'clinical' in self.data_modalities:
			self.clin_cat = clin_data_categorical.values.tolist()
			self.clin_cont = clin_data_continuous.values.tolist()

		# mRNA
		if 'mRNA' in self.data_modalities:
			data_mrna = dataframes['mRNATPM']
			self.data_mrna = data_mrna.values.tolist()

		# miRNA
		if 'miRNA' in self.data_modalities:
			data_mirna = dataframes['miRNA']
			self.data_mirna = data_mirna.values.tolist()

		# CNV
		if 'CNV' in self.data_modalities:
			data_cnv = dataframes['cnv']
			self.data_cnv = data_cnv.values.tolist()

	def __len__(self):
		return len(self.clin_cat)

	def __getitem__(self, index):
		data = {}
		data_label = {}
		target_y = np.array(self.target[index], dtype='int')
		target_y = torch.from_numpy(target_y)
		data_label['label'] = target_y.type(torch.LongTensor)


		if 'clinical' in self.data_modalities:
			clin_cate = np.array(self.clin_cat[index]).astype(np.int64)
			clin_cate = torch.from_numpy(clin_cate)
			data['clinical_categorical'] = clin_cate

			clin_conti = np.array(self.clin_cont[index]).astype(np.float32)
			clin_conti = torch.from_numpy(clin_conti)
			data['clinical_continuous'] = clin_conti


		if 'mRNA' in self.data_modalities:
			mrna = np.array(self.data_mrna[index])
			mrna = torch.from_numpy(mrna)
			data['mRNA'] = mrna.type(torch.FloatTensor)


		if 'miRNA' in self.data_modalities:
			mirna = np.array(self.data_mirna[index])
			mirna = torch.from_numpy(mirna)
			data['miRNA'] = mirna.type(torch.FloatTensor)

		if 'CNV' in self.data_modalities:
			cnv = np.array(self.data_cnv[index])
			cnv = torch.from_numpy(cnv)
			data['CNV'] = cnv.type(torch.FloatTensor)

		return data, data_label


# class MyDataset_test(torch.utils.data.Dataset):
# 	def __init__(self, modalities, dataframes, cate_cols):
# 		"""
# 		Parameters
# 		----------
# 		modalities: list
# 			Used modalities
#
# 		data_path: dict
# 			The path of used data.
#
# 		Returns
# 		-------
# 		data: dictionary
# 			{'clin_data_categorical': ,..,'mRNA': ...}
#
# 		data_label: dictionary
# 			{'label':[[event, time]]}
# 		"""
# 		super(MyDataset_test, self).__init__()
# 		self.data_modalities = modalities
# 		clin_data_categorical, clin_data_continuous = preprocess_clinical_data_test(dataframes, cate_cols)
#
# 		# clinical
# 		if 'clinical' in self.data_modalities:
# 			self.clin_cat = clin_data_categorical.values.tolist()
# 			self.clin_cont = clin_data_continuous.values.tolist()
#
# 		# mRNA
# 		if 'mRNA' in self.data_modalities:
# 			data_mrna = dataframes['mRNATPM']
# 			self.data_mrna = data_mrna.values.tolist()
#
# 		# miRNA
# 		if 'miRNA' in self.data_modalities:
# 			data_mirna = dataframes['miRNA']
# 			self.data_mirna = data_mirna.values.tolist()
#
# 		# CNV
# 		if 'CNV' in self.data_modalities:
# 			data_cnv = dataframes['cnv']
# 			self.data_cnv = data_cnv.values.tolist()
#
# 	def __len__(self):
# 		return len(self.clin_cat)
#
# 	def __getitem__(self, index):
# 		data = {}
# 		if 'clinical' in self.data_modalities:
# 			clin_cate = np.array(self.clin_cat[index]).astype(np.int64)
# 			clin_cate = torch.from_numpy(clin_cate)
# 			data['clinical_categorical'] = clin_cate
#
# 			clin_conti = np.array(self.clin_cont[index]).astype(np.float32)
# 			clin_conti = torch.from_numpy(clin_conti)
# 			data['clinical_continuous'] = clin_conti
#
#
# 		if 'mRNA' in self.data_modalities:
# 			mrna = np.array(self.data_mrna[index])
# 			mrna = torch.from_numpy(mrna)
# 			data['mRNA'] = mrna.type(torch.FloatTensor)
#
#
# 		if 'miRNA' in self.data_modalities:
# 			mirna = np.array(self.data_mirna[index])
# 			mirna = torch.from_numpy(mirna)
# 			data['miRNA'] = mirna.type(torch.FloatTensor)
#
# 		if 'CNV' in self.data_modalities:
# 			cnv = np.array(self.data_cnv[index])
# 			cnv = torch.from_numpy(cnv)
# 			data['CNV'] = cnv.type(torch.FloatTensor)
#
# 		return data
































