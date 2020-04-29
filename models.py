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
from torch import nn


class BaselineGAP(nn.Module):
    def __init__(self, embedding_size):
        super(BaselineGAP, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)
        
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
        spk_embedding = torch.flatten(x, 1)
        out = self.pretrained.fc(spk_embedding)
        return spk_embedding, out
    
#----------------------------------------------------------------------
    
class BaselineSAP(nn.Module):
    def __init__(self, embedding_size,speakers):
        super(BaselineSAP, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)
        self.avgpool = nn.AvgPool2d((4,4),stride=4)
        self.linear = nn.Linear(256,256)
        self.attention = self.parameter(256,1)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(256)
        self.fc= nn.Linear(embedding_size,speakers)
        
    def parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        p = self.avgpool(x)
        b,c,h,w = p.size()
        p = p.view(b,c,h*w).permute(0,2,1)
        h = torch.tanh(self.linear(p))
        w = torch.matmul(h, self.attention)
        s = F.softmax(w,dim=1)
        e = torch.sum(p*w,dim=1)
        o = torch.flatten(e,1)
        o = self.dropout(o)
        spk_embedding = self.bn(o)
        out = self.fc(spk_embedding)
        return spk_embedding, out
        
#---------------------------------------------------------------------- 

class BaselineMSAE(nn.Module):
    def __init__(self, embedding_size,speakers):
        super(BaselineMSAE, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)
        self.avgpool0 = nn.AvgPool2d((4,4),stride=4)
        self.linear0 = nn.Linear(32,32)
        self.attention0 = self.parameter(32,1)
        self.bn0 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool2d((4,4),stride=4)
        self.linear1 = nn.Linear(32,32)
        self.attention1 = self.parameter(32,1)
        self.bn1 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool2d((4,4),stride=4)
        self.linear2 = nn.Linear(64,64)
        self.attention2 = self.parameter(64,1)
        self.bn2 = nn.BatchNorm1d(64)
        self.avgpool3 = nn.AvgPool2d((4,4),stride=4)
        self.linear3 = nn.Linear(128,128)
        self.attention3 = self.parameter(128,1)
        self.bn3 = nn.BatchNorm1d(128)
        self.avgpool4 = nn.AvgPool2d((4,4),stride=4)
        self.linear4 = nn.Linear(256,256)
        self.attention4 = self.parameter(256,1)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc= nn.Linear(embedding_size,speakers)
        
    def parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        p0 = self.avgpool0(x)
        b,c,h,w = p0.size()
        p0 = p0.view(b,c,h*w).permute(0,2,1)
        h0 = torch.tanh(self.linear0(p0))
        w0 = torch.matmul(h0, self.attention0)
        s0 = F.softmax(w0,dim=1)
        e0 = torch.sum(p0*w0,dim=1)
        o0 = torch.flatten(e0,1)
        o0 = self.dropout(o0)
        o0 = self.bn0(o0)
        
        x = self.pretrained.layer1(x)
        p1 = self.avgpool1(x)
        b,c,h,w = p1.size()
        p1 = p1.view(b,c,h*w).permute(0,2,1)
        h1 = torch.tanh(self.linear1(p1))
        w1 = torch.matmul(h1, self.attention1)
        s1 = F.softmax(w1,dim=1)
        e1 = torch.sum(p1*w1,dim=1)
        o1 = torch.flatten(e1,1)
        o1 = self.dropout(o1)
        o1 = self.bn1(o1)
        
        x = self.pretrained.layer2(x)
        p2 = self.avgpool2(x)
        b,c,h,w = p2.size()
        p2 = p2.view(b,c,h*w).permute(0,2,1)
        h2 = torch.tanh(self.linear2(p2))
        w2 = torch.matmul(h2, self.attention2)
        s2 = F.softmax(w2,dim=1)
        e2 = torch.sum(p2*w2,dim=1)
        o2 = torch.flatten(e2,1)
        o2 = self.dropout(o2)
        o2 = self.bn2(o2)
        
        x = self.pretrained.layer3(x)
        p3 = self.avgpool3(x)
        b,c,h,w = p3.size()
        p3 = p3.view(b,c,h*w).permute(0,2,1)
        h3 = torch.tanh(self.linear3(p3))
        w3 = torch.matmul(h3, self.attention3)
        s3 = F.softmax(w3,dim=1)
        e3 = torch.sum(p3*w3,dim=1)
        o3 = torch.flatten(e3,1)
        o3 = self.dropout(o3)
        o3 = self.bn3(o3)
        
        x = self.pretrained.layer4(x)
        p4 = self.avgpool4(x)
        b,c,h,w = p4.size()
        p4 = p4.view(b,c,h*w).permute(0,2,1)
        h4 = torch.tanh(self.linear4(p4))
        w4 = torch.matmul(h4, self.attention4)
        s4 = F.softmax(w4,dim=1)
        e4 = torch.sum(p4*w4,dim=1)
        o4 = torch.flatten(e4,1)
        o4 = self.dropout(o4)
        o4 = self.bn4(o4)
        
        o1 = torch.cat((o1,o0),1)
        o2 = torch.cat((o2,o1),1)
        o3 = torch.cat((o3,o2),1)
        o4 = torch.cat((o4,o3),1)
        
        spk_embedding = o4
        out = self.fc(spk_embedding)
        return spk_embedding, out    
    
#----------------------------------------------------------------------

def l2_norm(x):
    x_size = x.size() 
    buffer = torch.pow(x,2)
    normp = torch.sum(buffer,1).add_(1e-10)
    norm = torch.sqrt(normp)
    _out = torch.div(x, norm.view(-1,1).expand_as(x))
    out = _out.view(x_size)
    out = out*10
    return out
    
class BaselineSAMAFRN(nn.Module):
    def __init__(self, embedding_size,speakers):
        super(BaselineSAMAFRN, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)
        self.dropout = nn.Dropout(0.5)
        
        self.avgpool0 = nn.AvgPool2d((4,4),stride=4)
        self.linear0 = nn.Linear(32,32)
        self.attention0 = self.parameter(32,1)
        self.bn0 = nn.BatchNorm1d(32)
        self.avgpool1 = nn.AvgPool2d((4,4),stride=4)
        self.linear1 = nn.Linear(32,32)
        self.attention1 = self.parameter(32,1)
        self.bn1 = nn.BatchNorm1d(32)
        self.avgpool2 = nn.AvgPool2d((4,4),stride=4)
        self.linear2 = nn.Linear(64,64)
        self.attention2 = self.parameter(64,1)
        self.bn2 = nn.BatchNorm1d(64)
        self.avgpool3 = nn.AvgPool2d((4,4),stride=4)
        self.linear3 = nn.Linear(128,128)
        self.attention3 = self.parameter(128,1)
        self.bn3 = nn.BatchNorm1d(128)
        self.avgpool4 = nn.AvgPool2d((4,4),stride=4)
        self.linear4 = nn.Linear(256,256)
        self.attention4 = self.parameter(256,1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.se0 = nn.Linear(512,int(round(512/8)))
        self.lrelu = nn.LeakyReLU(0.01)
        self.se1 = nn.Linear(int(round(512/8)),512)
        self.sigmoid = nn.Sigmoid()
        
        self.fc= nn.Linear(embedding_size,speakers)
        
    def parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        p0 = self.avgpool0(x)
        b,c,h,w = p0.size()
        p0 = p0.view(b,c,h*w).permute(0,2,1)
        h0 = torch.tanh(self.linear0(p0))
        w0 = torch.matmul(h0, self.attention0)
        s0 = F.softmax(w0,dim=1)
        e0 = torch.sum(p0*w0,dim=1)
        o0 = torch.flatten(e0,1)
        o0 = self.dropout(o0)
        o0 = self.bn0(o0)
        
        x = self.pretrained.layer1(x)
        p1 = self.avgpool1(x)
        b,c,h,w = p1.size()
        p1 = p1.view(b,c,h*w).permute(0,2,1)
        h1 = torch.tanh(self.linear1(p1))
        w1 = torch.matmul(h1, self.attention1)
        s1 = F.softmax(w1,dim=1)
        e1 = torch.sum(p1*w1,dim=1)
        o1 = torch.flatten(e1,1)
        o1 = self.dropout(o1)
        o1 = self.bn1(o1)
        
        x = self.pretrained.layer2(x)
        p2 = self.avgpool2(x)
        b,c,h,w = p2.size()
        p2 = p2.view(b,c,h*w).permute(0,2,1)
        h2 = torch.tanh(self.linear2(p2))
        w2 = torch.matmul(h2, self.attention2)
        s2 = F.softmax(w2,dim=1)
        e2 = torch.sum(p2*w2,dim=1)
        o2 = torch.flatten(e2,1)
        o2 = self.dropout(o2)
        o2 = self.bn2(o2)
        
        x = self.pretrained.layer3(x)
        p3 = self.avgpool3(x)
        b,c,h,w = p3.size()
        p3 = p3.view(b,c,h*w).permute(0,2,1)
        h3 = torch.tanh(self.linear3(p3))
        w3 = torch.matmul(h3, self.attention3)
        s3 = F.softmax(w3,dim=1)
        e3 = torch.sum(p3*w3,dim=1)
        o3 = torch.flatten(e3,1)
        o3 = self.dropout(o3)
        o3 = self.bn3(o3)
        
        x = self.pretrained.layer4(x)
        p4 = self.avgpool4(x)
        b,c,h,w = p4.size()
        p4 = p4.view(b,c,h*w).permute(0,2,1)
        h4 = torch.tanh(self.linear4(p4))
        w4 = torch.matmul(h4, self.attention4)
        s4 = F.softmax(w4,dim=1)
        e4 = torch.sum(p4*w4,dim=1)
        o4 = torch.flatten(e4,1)
        o4 = self.dropout(o4)
        o4 = self.bn4(o4)
        
        o1 = torch.cat((o1,o0),1)
        o2 = torch.cat((o2,o1),1)
        o3 = torch.cat((o3,o2),1)
        o4 = torch.cat((o4,o3),1)
        
        z0 = self.dropout(o4)
        z0 = self.se0(z0)
        z0 = self.lrelu(z0)
        z0 = self.se1(z0)
        z0 = self.sigmoid(z0)
        z0 = o4 * z0 
        spk_embedding = l2_norm(z0)

        out = self.fc(spk_embedding)
        return spk_embedding, out

#----------------------------------------------------------------------   
def attention(q,k,v):
    score = torch.matmul(q.transpose(0,1),k)/math.sqrt(k.size(1))
    score = F.softmax(score, dim=-1)
    out = torch.matmul(score,v.transpose(0,1))
    return out

def masking(target):
    #k is scaling vector for masking
    mask = np.triu(np.ones((target.size(0),target.size(1))), k=0)
    tmp = mask.ravel()
    np.random.shuffle(tmp)
    ran_mask = tmp.reshape((target.size(0),target.size(1)))
    ran_mask = torch.from_numpy(ran_mask).float().cuda()
    output = torch.mul(target, ran_mask)
    return output

class BaselineMCSAE(nn.Module):
    def __init__(self, embedding_size,speakers):
        super(BaselineMCSAE, self).__init__()
        self.pretrained = resnet.resnet34(pretrained=False)  
        self.tf0 = nn.Linear(32,32)
        self.tf1 = nn.Linear(32,32)
        self.tf2 = nn.Linear(64,64)
        self.tf3 = nn.Linear(128,128)
        
        self.fc0 = nn.Linear(512,512)
        self.fc1 = nn.Linear(512,512)
        self.fc2 = nn.Linear(512,512)
        self.last = nn.Linear(512,speakers)
        
        self.lrelu = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        p0 = F.adaptive_avg_pool2d(x,1)
        p0 = torch.squeeze(p0)
        p0 = p0.view(x.size(0), -1)

        x = self.pretrained.layer1(x)
        p1 = F.adaptive_avg_pool2d(x,1)
        p1 = torch.squeeze(p1)
        p1 = p1.view(x.size(0), -1)

        #==== attention block 1 ==== 
        sf1 = attention(self.tf0(masking(p0)),p1,p1)
        ss1 = attention(p1,self.tf0(masking(p0)),p0)
        z1 = torch.matmul(sf1,ss1.transpose(0,1))

        x = self.pretrained.layer2(x)
        p2 = F.adaptive_avg_pool2d(x,1)
        p2 = torch.squeeze(p2)
        p2 = p2.view(x.size(0), -1)

        #==== attention block 2 =====
        sf2 = attention(self.tf1(masking(p1)),p2,p2)
        ss2 = attention(p2,self.tf1(masking(p1)),p1)
        z2 = torch.matmul(sf2,ss2.transpose(0,1))

        x = self.pretrained.layer3(x)
        p3 = F.adaptive_avg_pool2d(x,1)
        p3 = torch.squeeze(p3)
        p3 = p3.view(x.size(0), -1)

        #==== attention block 3 ====
        sf3 = attention(self.tf2(masking(p2)),p3,p3)
        ss3 = attention(p3,self.tf2(masking(p2)),p2)
        z3 = torch.matmul(sf3,ss3.transpose(0,1))

        x = self.pretrained.layer4(x)
        p4 = F.adaptive_avg_pool2d(x,1)
        p4 = torch.squeeze(p4) 
        p4 = p4.view(x.size(0), -1)

        #==== attention block 4 ====
        sf4 = attention(self.tf3(masking(p3)),p4,p4)
        ss4 = attention(p4,self.tf3(masking(p3)),p3)
        z4 = torch.matmul(sf4,ss4.transpose(0,1))

        z1 = torch.matmul(p0, z1)
        z2 = torch.matmul(z1, z2)
        z3 = torch.matmul(z2, z3)
        z4 = torch.matmul(z3, z4)
        
        out = torch.cat((z4, p4), 1)
        out = self.lrelu(self.bn0(out))
        out = self.fc0(out)
        out = self.lrelu(self.bn1(out))
        out = self.fc1(out)
        out = self.lrelu(self.bn2(out))
        spk_embedding = self.fc2(out) 
        out = self.lrelu(self.bn3(spk_embedding))
        out = self.last(out)
        out = F.log_softmax(out, dim=-1)
        return spk_embedding, out