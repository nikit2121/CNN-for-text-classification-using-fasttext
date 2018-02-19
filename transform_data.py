import re
import logging
import numpy as np
import pandas as pd
from collections import Counter

def clean_str(s):
	"""Clean sentence"""
	s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " have", s)
	s = re.sub(r"n\'t", " not", s)
	s = re.sub(r"\'re", " are", s)
	s = re.sub(r"\'d", " had / would", s)
	s = re.sub(r"\'ll", " will", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r'\([^()]*\)','',s)
	s = re.sub(r'\([^()]*\)','',s)
	s= re.sub(r'\w*[0-9]\w*','',s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	s = re.sub(r'\S*(x{2,}|X{2,})\S*',"xxx", s)
	s = re.sub(r'[^\x00-\x7F]+', "", s)
	return s.strip().lower()

def load_data_and_labels(df,cols):
	"""Load sentences and labels"""
	selected = cols
	x_raw = df[selected[0]].apply(lambda x: clean_str(x)).tolist()
	return x_raw

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""Iterate the data batch by batch"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size)+1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

#if __name__ == '__main__':
	#input_file = '/home/nikit/Desktop/jnj_data/train_test.csv'
	#load_data_and_labels(input_file)
