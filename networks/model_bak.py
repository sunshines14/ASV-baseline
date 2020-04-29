#!/usr/bin/python
#-*- coding: utf-8 -*-

# Copyright 2020 Sogang University Auditory Intelligence Laboratory (Author: Soonshin Seo) 
#
# MIT License


import math
import torch
import torch.nn.functional as F
import numpy as np
import networks.resnet as resnet
import networks.se_resnet as se_resnet
from torch import nn


class BaselineSAP_bak(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineSAP_bak, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)
        self.bn = nn.BatchNorm1d(256)
        self.lrelu = nn.LeakyReLU(0.01)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256,256)
        self.fc2 = nn.Linear(256,1211)
        self.bn2 = nn.BatchNorm1d(1211)
        self.fc3 = nn.Linear(1211,embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        self.fc4 = nn.Linear(embedding_size,1211)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        x = self.pretrained.avgpool(x)
        x = torch.flatten(x, 1) # b x 256
        h = self.lrelu(self.fc1(x)) # b x 256
        u = x.unsqueeze(dim=1) # b x 1 x 256
        h = h.unsqueeze(dim=1) # b x 1 x 256
        # (b x 256 x 1) * (b x 1 x 256) -> b x 256 x 256
        s = torch.bmm(h.transpose(1, 2).contiguous(), u) 
        w = self.softmax(s) # norm to column
        #  (b x 1 x 256) * (b x 256 x 256) -> b x 1 x 256 
        #e = torch.bmm(x.unsqueeze(dim=1), w)
        e = torch.bmm(h, w) 
        e = e.squeeze(dim=1) # b x 256
        out = self.dropout(e)
        out = self.lrelu(self.bn2(self.fc2(out)))
        spk_embedding = self.fc3(out)
        out = self.dropout(self.lrelu(self.bn3(spk_embedding)))
        out = self.fc4(spk_embedding)
        return spk_embedding, out
    
#---------------------------------------------------------------------- 

class BaselineMSAE_bak(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineMSAE_bak, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        self.fc0 = nn.Linear(32,32)
        self.fc1 = nn.Linear(32,32)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(512,1211)
        self.fc6 = nn.Linear(1211,embedding_size)
        self.fc7 = nn.Linear(embedding_size,1211)
        self.bn1 = nn.BatchNorm1d(1211)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.lrelu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        p0 = self.pretrained.avgpool(x)
        p0 = torch.flatten(p0, 1)
        h0 = self.lrelu(self.fc0(p0)) 
        u0 = p0.unsqueeze(dim=1) 
        h0 = p0.unsqueeze(dim=1) 
        s0 = torch.bmm(h0.transpose(1, 2).contiguous(), u0) 
        w0 = self.softmax(s0)
        e0 = torch.bmm(h0, w0) 
        e0 = e0.squeeze(dim=1)
        o0 = self.dropout(e0)
        
        x = self.pretrained.layer1(x)
        p1 = self.pretrained.avgpool(x)
        p1 = torch.flatten(p1, 1)
        h1 = self.lrelu(self.fc1(p1)) 
        u1 = p1.unsqueeze(dim=1) 
        h1 = p1.unsqueeze(dim=1) 
        s1 = torch.bmm(h1.transpose(1, 2).contiguous(), u1) 
        w1 = self.softmax(s1)
        e1 = torch.bmm(h1, w1) 
        e1 = e1.squeeze(dim=1)
        o1 = self.dropout(e1)
        
        x = self.pretrained.layer2(x)
        p2 = self.pretrained.avgpool(x)
        p2 = torch.flatten(p2, 1)
        h2 = self.lrelu(self.fc2(p2)) 
        u2 = p2.unsqueeze(dim=1) 
        h2 = p2.unsqueeze(dim=1) 
        s2 = torch.bmm(h2.transpose(1, 2).contiguous(), u2) 
        w2 = self.softmax(s2)
        e2 = torch.bmm(h2, w2) 
        e2 = e2.squeeze(dim=1)
        o2 = self.dropout(e2)
        
        x = self.pretrained.layer3(x)
        p3 = self.pretrained.avgpool(x)
        p3 = torch.flatten(p3, 1)
        h3 = self.lrelu(self.fc3(p3)) 
        u3 = p3.unsqueeze(dim=1) 
        h3 = p3.unsqueeze(dim=1) 
        s3 = torch.bmm(h3.transpose(1, 2).contiguous(), u3) 
        w3 = self.softmax(s3)
        e3 = torch.bmm(h3, w3) 
        e3 = e3.squeeze(dim=1)
        o3 = self.dropout(e3)
        
        x = self.pretrained.layer4(x)
        p4 = self.pretrained.avgpool(x)
        p4 = torch.flatten(p4, 1)
        h4 = self.lrelu(self.fc4(p4)) 
        u4 = p4.unsqueeze(dim=1) 
        h4 = p4.unsqueeze(dim=1) 
        s4 = torch.bmm(h4.transpose(1, 2).contiguous(), u4) 
        w4 = self.softmax(s4)
        e4 = torch.bmm(h4, w4) 
        e4 = e4.squeeze(dim=1)
        o4 = self.dropout(e4)
        
        o1 = torch.cat((o1,o0),1)
        o2 = torch.cat((o2,o1),1)
        o3 = torch.cat((o3,o2),1)
        o4 = torch.cat((o4,o3),1)
        
        out = self.lrelu(self.bn1(self.fc5(o4)))
        spk_embedding = self.fc6(out)
        out = self.dropout(self.lrelu(self.bn2(spk_embedding)))
        out = self.fc7(out)
        return spk_embedding, out