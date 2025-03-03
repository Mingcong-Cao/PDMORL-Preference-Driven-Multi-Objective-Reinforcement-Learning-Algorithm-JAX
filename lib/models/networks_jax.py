from __future__ import absolute_import, division, print_function
from numpy.core.fromnumeric import reshape
import torch
# import torch.nn as nn
import torch.nn.functional as F
# from lib.utilities import common_utils
from torch.autograd import Variable

from flax import linen as nn
import jax
import jax.numpy as jnp  # JAX NumPy

from typing import Any, Callable, Sequence, Optional

############################################################################################
# Function to initialize weights
############################################################################################
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

class MO_DDQN(nn.Module):
    '''
        Multi-objective version of DDQN
    '''

    def __init__(self, args):
        super(MO_DDQN, self).__init__()


        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        
                   
        
        self.affine_in = nn.Linear(self.state_size + self.reward_size,self.hidden_size)
        self.affine = nn.Linear(self.hidden_size,self.hidden_size)
        self.affine_hid = common_utils.get_clones(self.affine, self._layer_N)
        del self.affine
        self.affine_out =  nn.Linear(self.hidden_size, self.action_size * self.reward_size)
        

        self.affine_in.apply(init_weights)
        self.affine_hid.apply(init_weights)
        self.affine_out.apply(init_weights)
        

    def forward(self, state, preference):
        
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.affine_in(x))
        for i in range(self._layer_N):
            x = F.relu(self.affine_hid[i](x))
        q = self.affine_out(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)
        

        return q


class Actor(nn.Module):
    state_size : int
    action_size : int
    reward_size : int
    _layer_N : int
    hidden_size : int
    max_action : jnp.ndarray
    _kernel_init : Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, state, preference):
        x = jnp.concatenate((state, preference), axis = 1)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(self.hidden_size, kernel_init=self._kernel_init)(x)
        x = nn.relu(x)
        for i in range(self._layer_N):
            x = nn.Dense(self.hidden_size, kernel_init=self._kernel_init)(x)
            x = nn.relu(x)
        # x = nn.Sequential([nn.Dense(self.hidden_size), nn.relu]*self._layer_N)(x)
        x = nn.Dense(self.action_size, kernel_init=self._kernel_init)(x)
        x = nn.tanh(x)
        return x * self.max_action
    

class Critic(nn.Module):
    state_size : int
    action_size : int
    reward_size : int
    _layer_N : int
    hidden_size : int
    max_action : jnp.ndarray
    _kernel_init : Callable = nn.initializers.xavier_normal()

    def setup(self):
        # Q1 architecture
        self.affine_in_1 = nn.Dense(self.hidden_size, kernel_init=self._kernel_init)
        self.affine_hid_1 = [nn.Dense(self.hidden_size, kernel_init=self._kernel_init) for _ in range(self._layer_N)]
        self.affine_out_1 = nn.Dense(self.reward_size, kernel_init=self._kernel_init)

        # Q2 architecture
        self.affine_in_2 = nn.Dense(self.hidden_size, kernel_init=self._kernel_init)
        self.affine_hid_2 = [nn.Dense(self.hidden_size, kernel_init=self._kernel_init) for _ in range(self._layer_N)]
        self.affine_out_2 = nn.Dense(self.reward_size, kernel_init=self._kernel_init)

    def __call__(self, state, preference, action):
        x = jnp.concatenate((state, preference, action), axis = 1)
        x = x.reshape(x.shape[0], -1)

        x1 = self.affine_in_1(x)
        x1 = nn.relu(x1)
        for layer in self.affine_hid_1:
            x1 = layer(x1)
            x1 = nn.relu(x1)
        q1 = self.affine_out_1(x1)

        x2 = self.affine_in_2(x)
        x2 = nn.relu(x2)
        for layer in self.affine_hid_2:
            x2 = layer(x2)
            x2 = nn.relu(x2)
        q2 = self.affine_out_2(x2)

        return q1, q2
    
    def Q1(self, state, preference, action):
        x = jnp.concatenate((state, preference, action), axis = 1)
        x = x.reshape(x.shape[0], -1)

        x1 = self.affine_in_1(x)
        x1 = nn.relu(x1)
        for layer in self.affine_hid_1:
            x1 = layer(x1)
            x1 = nn.relu(x1)
        q1 = self.affine_out_1(x1)

        return q1





class EnvelopeLinearCQN_default(nn.Module):
    '''
        Linear Controllable Q-Network, Envelope Version
    '''

    def __init__(self, args ):
        super(EnvelopeLinearCQN_default, self).__init__()


        self.args = args
        self.state_size = args.obs_shape
        self.action_size = args.action_shape
        self.reward_size = args.reward_size
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        
        
        self.affine1 = nn.Linear(self.state_size + self.reward_size,
                                    (self.state_size + self.reward_size) * 16)
        self.affine2 = nn.Linear((self.state_size + self.reward_size) * 16,
                                  (self.state_size + self.reward_size) * 32)
        self.affine3 = nn.Linear((self.state_size + self.reward_size) * 32,
                                  (self.state_size + self.reward_size) * 64)
        self.affine4 = nn.Linear((self.state_size + self.reward_size) * 64,
                                  (self.state_size + self.reward_size) * 32)
        self.affine5 = nn.Linear((self.state_size + self.reward_size) * 32,
                                  self.action_size * self.reward_size)
               
        
    def H(self, Q, w, s_num, w_num):
        
        use_cuda = self.args.cuda
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).type(LongTensor)
        reQ = Q.view(-1, self.action_size * self.reward_size
                     )[mask].view(-1, self.reward_size)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.action_size * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.action_size * w_num)
        inds = prod.max(1)[1]
        mask = BoolTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def H_(self, Q, w, s_num, w_num):
        
        use_cuda = self.args.cuda
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
        
        reQ = Q.view(-1, self.reward_size)

        # extend preference batch
        w_ext = w.unsqueeze(2).repeat(1, self.action_size, 1).view(-1, self.reward_size)

        # produce the inner products
        prod = torch.bmm(reQ.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions
        prod = prod.view(-1, self.action_size)
        inds = prod.max(1)[1]
        mask = BoolTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.reward_size)

        # get the HQ
        HQ = reQ.masked_select(Variable(mask)).view(-1, self.reward_size)

        return HQ

    def forward(self, state, preference, w_num=1):
        
        s_num = int(preference.size(0) / w_num)
        x = torch.cat((state, preference), dim=1)
        x = x.view(x.size(0), -1)

               
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))
        x = F.relu(self.affine4(x))
        q = self.affine5(x)
        q = q.view(q.size(0), self.action_size, self.reward_size)   
        

        hq = self.H_(q.detach().view(-1, self.reward_size), preference, s_num, w_num)     

        return q