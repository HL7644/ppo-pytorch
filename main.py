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

!pip install tensorboardX
from tensorboardX import SummaryWriter
writer=SummaryWriter(logdir='ppo') #can utilize tensorboard

from policy_module import *
from value_module import *
from discrete_agent import *

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#implemented w/ Cart-Pole environment

class Observation_Wrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(Observation_Wrapper, self).__init__(env)
  
  def observation(self, observation): #can apply tile coding
    return torch.FloatTensor(observation).to(device)

Episode=collections.namedtuple("Episode", field_names=['ep_obs','ep_aidx','ep_rew','ep_categ','ep_tde'])

class PPO():
  def __init__(self, agent, eps):
    self.agent=agent
    #hyperparameters
    self.eps=eps
  
  def get_ppo_policy_loss(self, batch_data):
    batch_size=len(batch_data)
    ppo_policy_loss=torch.FloatTensor([0]).to(device)
    for b_idx, episode in enumerate(batch_data):
      len_ep=episode.ep_obs.size(0)
      GAE=torch.FloatTensor([0]).to(device)
      ep_loss=torch.FloatTensor([0]).to(device)
      for step in range(len_ep-1,-1,-1):
        obs=episode.ep_obs[step]
        a_idx=episode.ep_aidx[step]
        GAE=(self.agent.gamma*self.agent.lambd)*GAE+episode.ep_tde[step].item() #GAE independent of pm
        categ=episode.ep_categ[step]
        old_prob=categ.probs[a_idx].item()
        new_categ=self.agent.pm(obs)
        new_prob=new_categ.probs[a_idx]
        p_ratio=new_prob/old_prob
        loss=min(p_ratio*GAE, torch.clamp(p_ratio, 1-self.eps, 1+self.eps)*GAE)
        ep_loss=ep_loss-loss #negative to perform gradient ascent
      ep_loss=ep_loss/len_ep
      ppo_policy_loss=ppo_policy_loss+ep_loss
    ppo_policy_loss=ppo_policy_loss/batch_size
    return ppo_policy_loss
  
  def get_value_loss(self, batch_data):
    batch_size=len(batch_data)
    value_loss=torch.FloatTensor([0]).to(device)
    for b_idx, episode in enumerate(batch_data):
      len_ep=len(episode)
      rtg=0
      ep_value_loss=torch.FloatTensor([0]).to(device)
      for step in range(len_ep, -1, -1):
        obs=episode.ep_obs[step]
        rtg=rtg+episode.ep_rew[step]
        V=self.agent.vm(obs)
        loss=(V-rtg)**2
        ep_value_loss=ep_value_loss+loss
      ep_value_loss=ep_value_loss/len_ep
      value_loss=value_loss+ep_value_loss
    value_loss=value_loss/batch_size
    return value_loss
  
  def cartpole_test(self, batch_data):
    batch_size=len(batch_data)
    avg_len_ep=0
    for episode in batch_data:
      len_ep=episode.ep_obs.size(0)
      avg_len_ep=avg_len_ep+len_ep
    avg_len_ep=avg_len_ep/batch_size
    return avg_len_ep
  
  def train(self, batch_size, n_epochs, n_p_epochs, n_v_epochs, p_lr, v_lr):
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)
    value_optim=optim.SGD(self.agent.vm.parameters(), lr=v_lr)
    for epoch in range(1, n_epochs+1):
      old_param_vector=self.agent.pm.vectorize_parameters()
      batch_data=self.agent.get_batch_data(batch_size=batch_size, custom_param=old_param_vector)

      old_policy_loss=0
      for p_epoch in range(1, n_p_epochs+1):
        ppo_policy_loss=self.get_ppo_policy_loss(batch_data)

        policy_optim.zero_grad()
        ppo_policy_loss.backward()
        policy_optim.step()
        writer.add_scalar("Policy Loss Epoch {:d}".format(epoch), ppo_policy_loss.item(), p_epoch)
        p_increment=torch.abs(ppo_policy_loss-old_policy_loss)
        if p_increment<1e-3: #incremental apporach instead of fixed epochs
          break
        old_policy_loss=ppo_policy_loss
      old_value_loss=0
      for v_epoch in range(1, n_v_epochs+1):
        value_loss=self.get_value_loss(batch_data)

        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        writer.add_scalar("Value Loss Epoch {:d}".format(epoch), value_loss.item(), v_epoch)
        v_increment=torch.abs(value_loss-old_value_loss)
        if v_increment<1e-3:
          break
        old_value_loss=value_loss
      #test performance
      avg_len_ep=self.cartpole_test(batch_data)
      writer.add_scalar("Avg Len Ep", avg_len_ep, epoch)
    return

cart_pole_env=gym.make('CartPole-v1')
tensor_cartpole=Observation_Wrapper(cart_pole_env)
agent=Discrete_On_Policy_Agent(tensor_cartpole, .99, .97)
ppo=PPO(agent, eps=.2)
ppo.train(256, 50, 50, 50, 2e-4, 5e-3)