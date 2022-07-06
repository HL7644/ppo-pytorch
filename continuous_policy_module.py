import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
from torch.distributions.normal import Normal
import random

import collections

import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mean_Module(nn.Module):
  def __init__(self, o_dim, a_dim):
    super(Mean_Module, self).__init__()
    self.linear1=nn.Linear(o_dim, 64).to(device)
    self.linear2=nn.Linear(64, a_dim).to(device)
    self.w_sizes, self.b_sizes=self.get_parameter_sizes()
  
  def get_parameter_sizes(self): #initialize linear layers in this part
    w_sizes=[]
    b_sizes=[]
    for element in self.children():
      if isinstance(element, nn.Linear):
        nn.init.xavier_uniform_(element.weight)
        nn.init.zeros_(element.bias)
        w_s=element.weight.size()
        b_s=element.bias.size()
        w_sizes.append(w_s)
        b_sizes.append(b_s)
    return w_sizes, b_sizes
  
  def vectorize_parameters(self):
    parameter_vector=torch.Tensor([]).to(device)
    for param in self.parameters():
      p=param.reshape(-1,1)
      parameter_vector=torch.cat((parameter_vector, p), dim=0)
    return parameter_vector.squeeze(dim=1)
  
  def inherit_parameters(self, parameter_vector):
    #size must be identical, input in vectorized form
    vector_idx=0
    #extract weight, bias data
    weights=[]
    biases=[]
    for sz_idx, w_size in enumerate(self.w_sizes):
      w_length=w_size[0]*w_size[1]
      weight=parameter_vector[vector_idx:vector_idx+w_length]
      weight=weight.reshape(w_size[0], w_size[1])
      weights.append(weight)
      vector_idx=vector_idx+w_length
      b_length=self.b_sizes[sz_idx][0]
      bias=parameter_vector[vector_idx:vector_idx+b_length]
      bias=bias.reshape(-1)
      biases.append(bias)
      vector_idx=vector_idx+b_length
    #overwrite parameters
    linear_idx=0
    for element in self.children():
      if isinstance(element, nn.Linear):
        element.weight=nn.Parameter(weights[linear_idx])
        element.bias=nn.Parameter(biases[linear_idx])
        linear_idx+=1
    return
  
  def forward(self, obs):
    tanh=nn.Tanh()
    layers=nn.Sequential(self.linear1, tanh, self.linear2, tanh)
    output=layers(obs)
    return output

class DGP(nn.Module): #continuous stochastic policy module: Diagonal Gaussian Policy
  def __init__(self, o_dim, a_dim):
    super(DGP, self).__init__()
    self.o_dim=o_dim
    self.a_dim=a_dim
    self.mm=Mean_Module(o_dim, a_dim)
    self.log_std=nn.Parameter(torch.full([a_dim], 0.1).to(device)) #non-neural network log-std parameter
  
  def vectorize_parameters(self):
    mean_param=self.mm.vectorize_parameters()
    log_std_param=torch.FloatTensor(self.log_std).to(device)
    return mean_param, log_std_param
  
  def inherit_parameters(self, mean_param, log_std_param):
    self.mm.inherit_parameters(mean_param)
    self.log_std=nn.Parameter(log_std_param)
    return
  
  def forward(self, obs): #return distribution
    #get action
    mean=self.mm(obs)
    std=torch.exp(self.log_std)
    #use torch.distributions.Normal
    dist=Normal(mean, std)
    return dist