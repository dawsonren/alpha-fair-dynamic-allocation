import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        # Policy network (actor)
        self.pi_fc1 = nn.Linear(obs_dim, hidden_size)
        self.pi_fc2 = nn.Linear(hidden_size, hidden_size)
        self.pi_fc3 = nn.Linear(hidden_size, hidden_size)
        self.pi_out = nn.Linear(hidden_size, act_dim)

        # Q-value networks (critics)
        self.q1_fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_fc3 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)

        self.q2_fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_fc3 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

    def pi(self, obs):
        pi_out = F.relu(self.pi_fc1(obs))
        pi_out = F.relu(self.pi_fc2(pi_out))
        pi_out = F.relu(self.pi_fc3(pi_out))
        pi_out = F.sigmoid(self.pi_out(pi_out))
        return pi_out, torch.zeros(obs.shape[0])

    def q1(self, obs, act):
        q1_input = torch.cat([obs, act], dim=-1)
        q1_out = F.relu(self.q1_fc1(q1_input))
        q1_out = F.relu(self.q1_fc2(q1_out))
        q1_out = F.relu(self.q1_fc3(q1_out))
        q1_value = self.q1_out(q1_out)
        return q1_value.squeeze(-1)

    def q2(self, obs, act):
        q2_input = torch.cat([obs, act], dim=-1)
        q2_out = F.relu(self.q2_fc1(q2_input))
        q2_out = F.relu(self.q2_fc2(q2_out))
        q2_out = F.relu(self.q2_fc3(q2_out))
        q2_value = self.q2_out(q2_out)
        return q2_value.squeeze(-1)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)[0].cpu().numpy()