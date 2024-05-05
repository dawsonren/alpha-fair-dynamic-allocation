import random
from collections import deque
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces
import numpy as np

from src.rl.actors import PowerActor, NNActor, GaussianPolicy
from src.rl.critic import Critic, QNetwork
from src.rl.utils import soft_update, hard_update

class ReplayBuffer:
    """
    Replay to overcome challenge of data not being independently distributed
    during on-policy training
    """
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, s, a, r, ss, done):
        """ state, action, reward, next_state """
        experience = (s, a, r, ss, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        s_batch = [experience[0] for experience in batch]
        a_batch = [experience[1] for experience in batch]
        r_batch = [experience[2] for experience in batch]
        ss_batch = [experience[3] for experience in batch]
        done_batch = [experience[4] for experience in batch]

        return s_batch, a_batch, r_batch, ss_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class SAC:
    def __init__(self,
                 N,
                 lr=1e-3,
                 gamma=0.99,
                 tau=1e-2,
                 update_interval=1,
                 target_update_interval=5e3,
                 hidden_size=256,
                 memory_max_size=50000):

        num_inputs = 2 + 2 * N
        action_shape = 1

        self.gamma = gamma
        self.tau = tau
        self.alpha = 0

        self.target_update_interval = target_update_interval
        self.update_interval = update_interval

        self.device = torch.device("cpu")

        self.critic = QNetwork(num_inputs, action_shape, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, action_shape, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(action_shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.policy = GaussianPolicy(num_inputs, action_shape, hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.rewards = []
        self.memory = ReplayBuffer(memory_max_size)
        self.updates = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, batch_size):
        for _ in range(self.update_interval):
            # get a sample from the replay buffer
            states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
            state_batch = torch.stack(states, dim=0)
            action_batch = torch.FloatTensor(np.array(actions))
            reward_batch = torch.FloatTensor(rewards)
            next_state_batch = torch.stack(next_states, dim=0)
            mask_batch = torch.FloatTensor(dones)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs

            self.updates += 1

            if self.updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

class DDPG:
    def __init__(
            self,
            N,
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            gamma=0.99,
            tau=1e-2,
            memory_max_size=50000
            ):

        ## params
        # state is (t, c_t, d_1, ..., d_N, x_1, ..., x_N)
        self.states_dim = 2 + 2 * N
        # action is a scalar
        self.actions_dim = 1
        self.gamma = gamma
        self.tau = tau

        ## networks
        # initialize critic and actor network
        self.actor = NNActor(self.states_dim, 1)
        self.critic = Critic(self.states_dim + self.actions_dim, 1)

        # initialize critic and actor target network
        self.actor_target = NNActor(self.states_dim, self.actions_dim)
        self.critic_target = Critic(self.states_dim + self.actions_dim, 1)

        # initialize both with the same parameters
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = ReplayBuffer(memory_max_size)
        self.loss = nn.MSELoss()

        self.rewards = []

    def select_action(self, state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state = state.float().unsqueeze(0)
        action = self.actor.forward(state)
        action = action.detach().numpy()[0]
        return action

    def update(self, batch_size):
        # get a sample from the replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.stack(states, dim=0)
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states, dim=0)

        # update critic by minimizing loss
        Q_vals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Q_prime = rewards + self.gamma * next_Q

        critic_loss = self.loss(Q_vals, Q_prime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor policy using sampled policy gradient
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks (slowly using soft updates/polyak averaging)
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)