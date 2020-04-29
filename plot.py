#!/usr/bin/python
#-*- coding: utf-8 -*-

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License

import os
import sys
import soundfile
import speechpy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import nn
from torchvision import transforms
from models import Baseline, BaselineSAP, BaselineMSAE
from sklearn.manifold import TSNE


def tsne(model, device, data_path):
    with open(data_path, 'r') as f:
        lines = f.read().splitlines()
    n = 1
    m = 0
    pre_speaker = '01'
    #pre_speaker = '10001'
    #cmap = cm.rainbow(np.linspace(0.0, 1.0, 30))
    cmap = ['blue', 'red', 'green', 'gold', 'rosybrown', 'firebrick', 
            'darkcyan', 'orange', 'purple', 'sandybrown', 'bisque', 
            'tan', 'm', 'darkkhaki', 'olivedrab', 'chartreuse', 
            'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 
            'rosybrown', 'slategray', 'royalblue', 'navy', 'black', 
            'darkorchid', 'm', 'darksalmon', 'crimson', 'sienna', 'gray']
    plt.figure()
    for line in lines:
        print (line)
        line = line.strip()
        embedding = get_embedding(model, device, line)
        line_splited = line.split('/')
        #speaker = line_splited[-3]
        #speaker = speaker.replace('id','')
        speaker = line_splited[-2]
        #embedding = embedding.view(128, 2)
        embedding = embedding.view(64, 2)
        embedding = embedding.cpu().detach().numpy()
        tsne = TSNE(n_components=2)
        e_tsne = tsne.fit_transform(embedding)
        if int(pre_speaker) != int(speaker):
            pre_speaker = speaker
            m = m+1
        #m = m+1
        #for i in range(len(e_tsne)):
            #e_min, e_max = np.min(e_tsne, 0), np.max(e_tsne, 0)
            #e_tsne = (e_tsne - e_min) / (e_max - e_min)
            #plt.scatter(e_tsne[i, 0], e_tsne[i, 1], s=5, color=cmap[m]) 
        plt.scatter(e_tsne[:,0], e_tsne[:,1], s=5, color=cmap[m])
        if n % 10 == 0:
            #plt.xlim(-15, 15)
            #plt.ylim(-15, 15)
            plt.xticks([]), plt.yticks([])
            #plt.show()
            plt.savefig('tools/result/pic.png', dpi=100)      
        n = n+1
    #plt.xlim(-30, 30)
    #plt.ylim(-30, 30)
    plt.xticks([]), plt.yticks([])
    plt.savefig('tools/result/pic.png', dpi=100)
    
def plot(model, device, data_path):
    n = 0
    embedding_list = []
    with open(data_path, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        print (line)
        line = line.strip()
        line_splited = line.split('/')
        #speaker = line_splited[-3]
        speaker = line_splited[-2]
        
        embedding = get_embedding(model, device, line)
        embedding_list.append(embedding)
        embedding = embedding.view(16, 8)
        embedding = embedding.cpu().detach().numpy()
        
        fig, ax1 = plt.subplots(1,1)
        im = ax1.imshow(embedding, interpolation='nearest')
        cb = plt.colorbar(im)
        fg_color = 'black'
        bg_color = 'black'
        ax1.set_title(speaker, color=fg_color)
        ax1.patch.set_facecolor(bg_color)
        im.axes.tick_params(color=fg_color, labelcolor=fg_color)
        
        for spine in im.axes.spines.values():
            spine.set_edgecolor(fg_color)    
        #cb.set_label('scale', color=fg_color)
        cb.ax.yaxis.set_tick_params(color=fg_color)
        cb.outline.set_edgecolor(fg_color)
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)
        fig.patch.set_facecolor(bg_color)    
        plt.tight_layout()
        plt.savefig('tools/result/plot_{}.png'.format(n, dpi=100))
        n += 1
    
    score = F.cosine_similarity(embedding_list[0], embedding_list[1])
    score = score.cpu().detach().numpy()[0]
    print('\n{:.5f}'.format(score))
        
def get_embedding(model, device, pair_path):
    # +) test feature (online)
    y, sr = soundfile.read(pair_path)
    temp = get_feature(y)
    temp = torch.Tensor(temp)
    test_x = temp.unsqueeze(dim=0)
    # +) embedding
    model.eval()
    test_x = test_x.to(device)
    spk_emb, _ = model(test_x)
    # +) l2-norm
    spk_emb_norm = l2_norm(spk_emb)
    return spk_emb_norm
        
def get_feature(x):
    fbank = speechpy.feature.lmfe(
        x, sampling_frequency=16000, frame_length=0.025, frame_stride=0.01, 
        num_filters=64, fft_length=512, low_frequency=0, high_frequency=None)
    fbank_norm = speechpy.processing.cmvnw(fbank, win_size=301, variance_normalization=True)
    return fbank_norm 
            
def l2_norm(x):
    x_size = x.size() 
    buffer = torch.pow(x, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _out = torch.div(x, norm.view(-1, 1).expand_as(x))
    out = _out.view(x_size)
    out = out * 10
    return out

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = BaselineSAP(128).to(device)
    test_checkpoint_path = 'models/model_vox1_0_fbank_64_96_200_128_baselineSAP07/epoch_92.pth'
    model.load_state_dict(torch.load(test_checkpoint_path))
    data_path = 'tools/scp/test_06_01.scp'
    #tsne(model, device, data_path)
    plot(model, device, data_path)