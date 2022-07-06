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

!pip install tensorboardX
from tensorboardX import SummaryWriter
writer=SummaryWriter(logdir='ppo') #can utilize tensorboard

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from continuous_policy_module import *
from value_module import *

Ep_Step=collections.namedtuple('EpStep', field_names=['obs', 'action','reward','obs_f','termin_signal'])

class Agent():
  def __init__(self, env, test_env, gamma, lambd):
    self.env=env
    self.test_env=test_env
    self.gamma=gamma
    self.lambd=lambd
    
    self.o_dim=self.env.observation_space.shape[0]
    self.a_dim=self.env.action_space.shape[0]
    self.a_low=torch.FloatTensor(self.env.action_space.low).to(device)
    self.a_high=torch.FloatTensor(self.env.action_space.high).to(device)

    #set policy modules
    self.pm=DGP(self.o_dim, self.a_dim)
    pm_mv, pm_lstd=self.pm.vectorize_parameters()
    self.target_pm=DGP(self.o_dim, self.a_dim)
    self.target_pm.inherit_parameters(pm_mv, pm_lstd)

    #set value modules
    self.vm=Value_Module(self.o_dim, [400,300])
  
  def episode_generator(self): #generate ep w.r.t current policy
    ep_data=[]
    obs=self.env.reset()
    while True:
      dist=self.pm(obs)
      action=dist.sample()
      #clamp action
      clamped_action=torch.clamp(action, self.a_low, self.a_high).detach().cpu().numpy()
      #clamp action for progression
      obs_f, reward, termin_signal, _=self.env.step(clamped_action)
      if termin_signal and obs_f[0]>=0.45:
        rts=1 #real termin signal: not just reaching horizon
      else:
        rts=0
      #store actual action
      ep_step=Ep_Step(obs, action, reward, obs_f, rts)
      ep_data.append(ep_step)
      if termin_signal:
        break
    return ep_data
  
  def get_batch_data(self, batch_size):
    batch_data=[]
    for b_idx in range(batch_size):
      ep_data=self.episode_generator()
      batch_data.append(ep_data)
    return batch_data

class PPO():
  def __init__(self, agent):
    self.agent=agent
  
  def check_performance(self):
    avg_return=0
    avg_len_ep=0
    for n in range(1,10+1):
      acc_rew=0
      len_ep=0
      obs=self.test_env.reset()
      while True:
        dist=self.agent.pm(obs)
        action=dist.sample()
        action=torch.clamp(action, self.agent.a_low, self.agent.a_high).detach().cpu().numpy()
        obs_f, reward, termin_signal,_=self.agent.test_env.step(action)
        acc_rew+=reward
        len_ep+=1
        if termin_signal:
          break
      avg_return=avg_return+(acc_rew-avg_return)/n
      avg_len_ep=avg_len_ep+(len_ep-avg_len_ep)/n
    return avg_return, avg_len_ep
    
  def get_policy_loss(self, batch_data, clip_eps):
    policy_loss=torch.FloatTensor([0]).to(device)
    #PPO-Clip version
    batch_size=len(batch_data)
    for ep_data in batch_data:
      ep_loss=torch.FloatTensor([0]).to(device)
      GAE=torch.FloatTensor([0]).to(device) #general advantage estimator
      len_ep=len(ep_data)
      for step in range(len_ep-1, -1, -1): #reversing order
        ep_step=ep_data[step]
        obs=ep_step.obs
        V=self.agent.vm(obs)
        reward=ep_step.reward
        obs_f=ep_step.obs_f
        V_f=self.agent.vm(obs_f)
        #recursive GAE
        tde=reward+self.agent.gamma*V_f-V
        GAE=GAE*(self.agent.gamma*self.agent.lambd)+tde
        #probability
        dist=self.agent.pm(obs)
        action=ep_step.action
        log_prob=dist.log_prob(action)
        prob=torch.exp(log_prob)
        #old probability of current action
        old_dist=self.agent.target_pm(obs)
        old_log_prob=old_dist.log_prob(action)
        old_prob=torch.exp(old_log_prob).detach() #remove gradient for target values
        p_ratio=torch.div(prob, old_prob)

        #get PPO loss
        loss=min(GAE*p_ratio, torch.clamp(p_ratio, 1-clip_eps, 1+clip_eps)*GAE)
        ep_loss=ep_loss-loss #negative for GA
      policy_loss=policy_loss+ep_loss/len_ep
    policy_loss=policy_loss/batch_size

    return policy_loss
  
  def get_value_loss(self, batch_data):
    value_loss=torch.FloatTensor([0]).to(device)
    #MSE Loss w.r.t RTGs
    batch_size=len(batch_data)
    for ep_data in batch_data:
      ep_loss=torch.FloatTensor([0]).to(device)
      len_ep=len(ep_data)
      #final ep_step
      final_step=ep_data[len_ep-1]
      rts=final_step.termin_signal
      if rts:
        rtg=0
      else:
        final_obs=final_step.obs_f
        rtg=self.agent.vm(final_obs) #targets, use discounted rtgs, bootstrap due to reaching horizon
      for step in range(len_ep-1, -1, -1):
        ep_step=ep_data[step]
        rtg=rtg+ep_step.reward
        obs=ep_step.obs
        V=self.agent.vm(obs)
        ep_loss=ep_loss+(V-rtg)**2
      ep_loss=ep_loss/len_ep
      value_loss=value_loss+ep_loss
    value_loss=value_loss/batch_size

    return value_loss
  
  def train(self, batch_size, n_epochs, n_p_epochs, n_v_epochs, clip_eps, p_lr, v_lr):
    policy_optim=optim.Adam(self.agent.pm.parameters(), lr=p_lr)
    value_optim=optim.SGD(self.agent.vm.parameters(), lr=v_lr)

    for epoch in range(1, n_epochs+1):
      batch_data=self.agent.get_batch_data(batch_size)
      for p_epoch in range(1, n_p_epochs+1):
        policy_loss=self.get_policy_loss(batch_data, clip_eps)
        
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()
        writer.add_scalar("Policy Loss Epoch {:d}".format(epoch), policy_loss.item(), p_epoch)

      for v_epoch in range(1, n_v_epochs+1):
        value_loss=self.get_value_loss(batch_data)

        value_optim.zero_grad()
        value_loss.backward()
        value_optim.step()
        writer.add_scalar("Value Loss Epoch {:d}".format(epoch), value_loss.item(), p_epoch)

      #inherit paramters for old policy probability
      m_pv, lstd=self.agent.pm.vectorize_paramters()
      self.agent.target_pm.inherit_parameters(m_pv, lstd)
      #check performance every epoch
      avg_return, avg_len_ep=self.check_performance()
      writer.add_scalar("Avg Return", avg_return, epoch)
      writer.add_scalar("Avg Ep Length", avg_len_ep, epoch)
      print("Epoch: {:d}, Avg Return: {:.3f}, Avg Ep Length: {:.2f}".format(epoch, avg_return, avg_len_ep))
    return