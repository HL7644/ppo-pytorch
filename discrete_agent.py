import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as dsets
import numpy as np
from torch.distributions.categorical import Categorical
import random

import collections

import gym

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from policy_module import *
from value_module import *

Episode=collections.namedtuple("Episode", field_names=['ep_obs','ep_aidx','ep_rew','ep_categ','ep_tde'])

class Discrete_On_Policy_Agent():
  def __init__(self, env, gamma, lambd):
    self.env=env
    i_size=self.env.observation_space.shape[0]
    o_size=self.env.action_space.n
    self.pm=Policy_Module(i_size, [64,64], o_size)
    self.vm=Value_Module(i_size, [64,64])
    self.gamma=gamma
    self.lambd=lambd
  
  def get_old_policy(self, old_param, obs):
    old_logits=self.pm.parameter_forward(obs, old_param)
    old_categ=Categorical(logits=old_logits)
    return old_categ

  def episode_generator(self, horizon, custom_param): #on policy & can choose paramerters
    obs=self.env.reset()
    ep_obs=torch.Tensor([]).to(device)
    ep_obs=torch.cat((ep_obs, obs.unsqueeze(dim=0)), dim=0)
    ep_aidx=[]
    ep_reward=[]
    ep_categorical=[]
    ep_tde=[]
    ep_step=1
    while True:
      if custom_param is not None:
        categ=self.get_old_policy(custom_param, obs)
      else:
        categ=self.pm(obs)
      a_idx=categ.sample()
      ep_aidx.append(a_idx)
      ep_categorical.append(categ)
      obs_f, reward, termin_signal, _=self.env.step(a_idx.item())
      V=self.vm(obs)
      V_f=self.vm(obs_f)
      if termin_signal:
        reward=-100
        ep_reward.append(reward)
        tde=reward-V
        ep_tde.append(tde)
        break
      elif ep_step==horizon:
        ep_reward.append(reward)
        tde=reward+self.gamma*V_f-V
        ep_tde.append(tde)
        break
      else:
        ep_reward.append(reward)
        tde=reward+self.gamma*V_f-V
        ep_tde.append(tde)
        ep_obs=torch.cat((ep_obs, obs_f.unsqueeze(dim=0)), dim=0)
        obs=obs_f
        ep_step=ep_step+1
    ep_reward=torch.FloatTensor(ep_reward).to(device)
    episode=Episode(ep_obs, ep_aidx, ep_reward, ep_categorical, ep_tde)
    return episode
  
  def get_batch_data(self, batch_size, horizon=np.inf, custom_param=None):
    batch_data=[]
    for _ in range(batch_size):
      ep=self.episode_generator(horizon=horizon, custom_param=custom_param)
      batch_data.append(ep)
    return batch_data