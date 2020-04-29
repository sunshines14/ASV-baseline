#!/usr/bin/python
#-*- coding: utf-8 -*-

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import os
import collections
import soundfile
import librosa
import speechpy
import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from joblib import Parallel, delayed
from random import randint
from random import seed


jobs = 8
data_root = '/home/CORPUS/VoxCeleb'
meta_file = collections.namedtuple('meta_file', ['spk_id', 'path'])

class Dataset(Dataset):
    def __init__(self, version=None, data=None, size=None, feature=None, dims=None): 
        # 1) data dir
        self.data_dir = os.path.join(data_root,'{}/{}'.format(version, data))
        
        # 2) spk id int
        self.spk_id_dict = sorted(set(os.listdir(self.data_dir)))
        self.spk_id_int = {spk:i for i,spk in enumerate(self.spk_id_dict)}
        
        # 3) wav scp file
        self.wav_scp_file = os.path.join(data_root, '{}/meta/{}_{}_wav_all.scp'.format(version, version, data))
        print('wav scp file is ', self.wav_scp_file)
        
        # 4) cache file
        cache_file = '/home/CORPUS/VoxCeleb/features/cache_{}_{}_{}_{}_{}_all.npy'.format(version, data, size, feature, dims)
        print('cache file is ', cache_file)
        
        self.size = size
        self.feature = feature
        
        if os.path.exists(cache_file):
            self.data_x, self.data_y = torch.load(cache_file)
            print('dataset loaded from ', cache_file)
        else:
            data_meta = self.parse_wav_scp_file()
            temp = list(map(self.read_file, data_meta))
            self.data_x, self.data_y = map(list, zip(*temp))
            
            # 5) transforms
            self.transforms = transforms.Compose([
                lambda x: self.get_feature(x),
                lambda x: Tensor(x)])
            
            print('transforms ...')
            self.data_x = Parallel(n_jobs=jobs, prefer='threads')(delayed(self.transforms)(x) for x in self.data_x)
            torch.save((self.data_x, self.data_y), cache_file)
            print('dataset saved to ', cache_file)
            
        # 6) len
        self.length = len(self.data_x)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        padded_x = self.pad(x)
        return padded_x, y
    
    def parse_wav_scp_file(self):
        lines = [line.strip() for line in open(self.wav_scp_file).readlines()]
        data_meta = map(self.parse_line, lines)
        return list(data_meta)    

    def parse_line(self, lines):
        tokens = lines.split('/')
        return meta_file(
            spk_id = tokens[-3],
            path = lines)
    
    def read_file(self, data_meta):
        data_x, sr = soundfile.read(data_meta.path)
        idx = data_meta.spk_id
        data_y = self.spk_id_int[idx]
        return data_x, float(data_y), data_meta
    
    def pad(self, x):
        max_len = 1200
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        else:
            num_repeats = round((max_len/x_len) + 1)
            x_repeat = x.repeat(num_repeats, 1)
            padded_x = x_repeat[:max_len]
            return padded_x
    
    def get_feature(self, x):
        if self.feature == 'fbank':
            fbank = speechpy.feature.lmfe(
                x, sampling_frequency=16000, frame_length=0.025, frame_stride=0.01, 
                num_filters=64, fft_length=512, low_frequency=0, high_frequency=None)
            fbank_norm = speechpy.processing.cmvnw(fbank, win_size=301, variance_normalization=True)
        return fbank_norm

#if __name__ == '__main__':
#    loader = Dataset(version='vox2', data='train', size=0, feature='fbank', dims=64)