#!/usr/bin/python
#-*- coding: utf-8 -*-

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import os
import sys
import argparse
import soundfile
import speechpy
import librosa
import data_utils
import numpy as np
import torch
import torch.nn.functional as F
import config
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from tensorboardX import SummaryWriter
from models import BaselineGAP, BaselineSAP, BaselineMSAE, BaselineSAMAFRN, BaselineMCSAE
from tools.pytorchtools import EarlyStopping
from torchsummary import summary
from torch.autograd import Function
from scipy import linalg


def train(train_loader, model, device, criterion, optim):
    batch_idx = 0
    num_total = 0
    running_loss = 0
    running_correct = 0
    epoch_loss = 0
    epoch_acc = 0
    # +) train mode (parallel) 
    if device == 'cuda':
        model = nn.DataParallel(model).train()
    else:
        model.train()
    for batch_x, batch_y in train_loader:
        batch_idx += 1
        num_total += batch_x.size(0)
        # +) wrapping
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # +) forward 
        _, batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        if args.loss == 'orth':   
            # criterion is a list composed of crossentropy loss and OLE loss.
            losses_list = [-1, -1]
            # output_Var contains scores in the first element and features in the second element
            batch_loss = 0
            for cix, crit in enumerate(criterion):
                if cix == 1:
                    losses_list[cix] = crit(batch_out, batch_y)[0]
                else:
                    losses_list[cix] = crit(batch_out, batch_y)
                batch_loss += losses_list[cix]
        else:
            batch_loss = criterion(batch_out, batch_y)
        # +) accmulate loss stats
        running_loss += (batch_loss.item()*batch_x.size(0))
        # +) accumulate accuracy stats
        running_correct += (batch_pred == batch_y).sum(dim=0).item()
        # +) print
        if batch_idx % 10 == 0:
            sys.stdout.write('\r \t {:.5f} {:.5f}'.format(running_correct/num_total, running_loss/num_total))
        # +) zero gradient
        optim.zero_grad()
        # +) backward
        batch_loss.backward()
        # +) update
        optim.step()
    epoch_loss = running_loss/num_total
    epoch_acc = running_correct/num_total
    return epoch_loss, epoch_acc

def valid(valid_loader, model, device, criterion):
    num_total = 0
    running_correct = 0
    running_loss = 0
    epoch_loss = 0
    epoch_acc = 0
    # +) valid mode						 
    model.eval()
    for batch_x, batch_y in valid_loader:
        num_total += batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        batch_loss = criterion(batch_out, batch_y)
        running_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item()*batch_x.size(0))
    epoch_loss = running_loss/num_total
    epoch_acc = running_correct/num_total
    return epoch_loss, epoch_acc

def test(model, device, test_out_path):
    data_root = '/home/CORPUS/VoxCeleb/'
    # +) protocol
    with open('protocols/'+args.test_protocol, 'r') as f:
        protocol = f.read().splitlines()
    # +) compute score
    for pair in protocol:
        pair_splited = pair.split(' ')
        pair1_path = data_root + '{}/{}/{}'.format(args.version, 'test', pair_splited[1])
        #pair1_path = pair_splited[1]
        pair1_out = get_embedding(model, device, pair1_path)
        pair2_path = data_root + '{}/{}/{}'.format(args.version, 'test', pair_splited[2])
        #pair2_path = pair_splited[2]
        pair2_out = get_embedding(model, device, pair2_path)
        score = F.cosine_similarity(pair1_out, pair2_out)
        score = score.cpu().detach().numpy()[0]
        with open(test_out_path, 'a') as f:
            f.write('{:.5f} {} {}\n'.format(score, pair_splited[1], pair_splited[2]))
        #print('\n{} - {} - {}'.format(score, pair_splited[1], pair_splited[2]))
        
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
    #spk_emb_norm = l2_norm(spk_emb)
    #return spk_emb_norm
    return spk_emb
        
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

class OLELoss(Function):
    def __init__(self, lambda_=0.1):
        super(OLELoss, self).__init__()
        self.lambda_ = lambda_
        self.dX = 0

    def forward(self, x, y):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        classes = np.unique(y)
        n, d = x.shape
        lambda_ = 1.
        delta = 1.
        # gradients initialization
        obj_c = 0
        dx_c = np.zeros((n, d))
        eigthd = 1e-6  # threshold small eigenvalues for a better subgradient
        # compute objective and gradient for first term \sum ||TX_c||*
        for c in classes:
            a = x[y == c, :]
            # SVD
            u, s, v = linalg.svd(a, full_matrices=False)
            v = v.T
            nuclear = np.sum(s)
            # L_c = max(DELTA, ||TY_c||_*)-DELTA
            if nuclear > delta:
                obj_c += nuclear
                # discard small singular values
                r = np.sum(s < eigthd)
                uprod = u[:, 0:u.shape[1] - r].dot(v[:, 0:v.shape[1] - r].T)
                dx_c[y == c, :] += uprod
            else:
                obj_c += delta
        # compute objective and gradient for secon term ||TX||*
        u, s, v = linalg.svd(x, full_matrices=False)  # all classes
        v = v.T
        obj_all = np.sum(s)
        r = np.sum(s < eigthd)
        uprod = u[:, 0:u.shape[1] - r].dot(v[:, 0:v.shape[1] - r].T)
        dx_all = uprod
        obj = (obj_c - lambda_ * obj_all) / n * np.float(self.lambda_)
        dx = (dx_c - lambda_ * dx_all) / n * np.float(self.lambda_)
        self.dX = torch.FloatTensor(dx).cuda()
        return torch.FloatTensor([float(obj)]).cuda()

    def backward(self, grad_output):
        dx = self.dX
        return dx, None

if __name__ == '__main__':
    # 1) parser
    parser = argparse.ArgumentParser()
    # +) data preparation
    parser.add_argument('--version', type=str, default='vox1')
    parser.add_argument('--win_size', type=int, default=0)
    parser.add_argument('--feature', type=str, default='fbank')
    parser.add_argument('--feature_dims', type=str, default=64) 
    # +) training
    parser.add_argument('--train_batch_size', type=int, default=96)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    #parser.add_argument('--es_patience', type=int, default=0)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--speakers', type=int, default=5994)
    parser.add_argument('--model_comment', type=str, default='baseline')
    # +) optimizer
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--sched_factor', type=float, default=0.1)
    parser.add_argument('--sched_patience', type=int, default=0)
    parser.add_argument('--sched_min_lr', type=float, default=0.001)
    # +) test
    parser.add_argument('--test_mode', action='store_true', default=False) 
    #parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_num_checkpoint', type=int, default=None)
    parser.add_argument('--test_protocol', type=str, default='protocol_vox1.txt')
    # +) resume
    parser.add_argument('--resume_mode', action='store_true', default=False)
    parser.add_argument('--resume_num_checkpoint', type=int, default=None) 
    # +) retrain
    args = parser.parse_args()
	
    # 2) model tag
    model_tag = 'model_{}_{}_{}_{}_{}_{}_{}'.format(
            args.version, args.win_size, args.feature, args.feature_dims, 
            args.train_batch_size, args.num_epochs, args.embedding_size)
    if args.model_comment:
        model_tag = model_tag + '_{}'.format(args.model_comment)
    print('model tag is ', model_tag)	
    
    # 3) model save path
    if not os.path.exists('models'):
        os.mkdir('models')
    model_save_path = os.path.join('models', model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    print('model save path is ', model_save_path)
						
    # 4) use cuda
    if torch.cuda.is_available():
        device = 'cuda'
        print('device is ', device)
    else:
        device = 'cpu'
        print('device is ', device)
        
    # 5) test
    if args.test_mode:
        # +) test dataset
        #print('========== test dataset ==========')
        #test_dataset = data_utils.Dataset(
        #        version=args.version, data='test', size=args.win_size, feature=args.feature, dims=args.feature_dims)
        #test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
        # +) load model
        print('========== test process ==========')
        model = BaselineSAMAFRN(args.embedding_size,args.speakers).to(device)
        #summary(model, input_size=(1200,64))
        test_checkpoint_path = '{}/epoch_{}.pth'.format(model_save_path, str(args.test_num_checkpoint))
        model.load_state_dict(torch.load(test_checkpoint_path))
        print('model loaded from ', test_checkpoint_path)
        # +) result
        test_out_path = '{}/{}.result'.format(model_save_path, str(args.test_num_checkpoint))
        test(model, device, test_out_path)
        print('test output saved to ', test_out_path)
    
    # 6) train & valid
    else:
        # +) valid dataset
        #print('========== valid dataset ==========')
        #valid_dataset = data_utils.Dataset(
        #        version=args.version, data='valid', size=args.win_size, feature=args.feature, dims=args.feature_dims)
        #valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=8)
        # +) train dataset
        print('========== train dataset ==========')
        train_dataset = data_utils.Dataset(
                version=args.version, data='train', size=args.win_size, feature=args.feature, dims=args.feature_dims)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)
        print('========== train process ==========')
        # +) model init (check resume mode)
        if args.resume_mode:
            model = BaselineSAMAFRN(args.embedding_size,args.speakers).to(device)
            summary(model, input_size=(1200,64))
            resume_checkpoint_path = '{}/epoch_{}.pth'.format(model_save_path, str(args.resume_num_checkpoint))
            model.load_state_dict(torch.load(resume_checkpoint_path))
            print('model for resume loaded from ', resume_checkpoint_path)
            start = args.resume_num_checkpoint+1
        # +) model init
        else:
            model = BaselineSAMAFRN(args.embedding_size,args.speakers).to(device)
            summary(model, input_size=(1200,64))
            start = 1  
        # +) loss
        if args.loss == 'nll':
            criterion = nn.NLLLoss()
        elif args.loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif args.loss =='orth':
            criterion = [nn.CrossEntropyLoss()] + [OLELoss()]
        # +) optimizer
        if args.optim == 'adam':
            optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optim == 'sgd':
            optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, dampening=0, weight_decay=args.wd)
        elif args.optim == 'rms': 
            optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        elif args.optim == 'ada': 
            optim = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # +) scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, 'min', factor=args.sched_factor, patience=args.sched_patience, min_lr=args.sched_min_lr, verbose=True)
        # +) early stopping
        #early_stopping = EarlyStopping(patience=args.es_patience, verbose=False) 
        # +) tensorboardX, log
        if not os.path.exists('logs'):
            os.mkdir('logs')
        writer = SummaryWriter('logs/{}'.format(model_tag))
        train_losses = []
        #valid_losses = []
        for epoch in range(start, start+args.num_epochs):
            train_loss, train_acc = train(train_loader, model, device, criterion, optim)
            #valid_loss, valid_acc = valid(valid_loader, model, device, criterion)
            writer.add_scalar('train_acc', train_acc, epoch)
            #writer.add_scalar('valid_acc', valid_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            #writer.add_scalar('valid_loss', valid_loss, epoch)
            print('\n{} - train acc: {:.5f} - train loss: {:.5f}'.format(epoch, train_acc, train_loss))
            #print('\n{} - train acc: {:.5f} - valid acc: {:.5f} - train loss: {:.5f} - valid loss: {:.5f}'.format(
            #    epoch, train_acc, valid_acc, train_loss, valid_loss))
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
            train_losses.append(train_loss)
            #valid_losses.append(valid_loss)
            #early_stopping(valid_loss, model)
            #if early_stopping.early_stop:
            #    print('early stopping !')
            #    break
            scheduler.step(train_loss)
            #scheduler.step(valid_loss)
        minposs = train_losses.index(min(train_losses))+1
        #minposs = valid_losses.index(min(valid_losses))+1
        print('lowest train loss at epoch is {}'.format(minposs))