"""
Create parametrized policies for our allocation function.

These are parametrized x(s_t; \theta) such that
we optimize over \theta as our agent acts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.rl.utils import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class PowerActor(nn.Module):
    """
    This policy has three parameters:
    \theta = (\beta_1, \beta_2, \beta_3)

    The input tensor is is s_t = (t, c_t, d_1, ..., d_n, x_1, ..., x_n).
    We use t to get the correct values for d_1, ..., d_i, x_1, ..., x_{i-1}.
    
    This represents the function
    u(s_t; \theta) = (d_t ** \beta_1 / (d_t ** \beta_2 + E[\sum_{j=t+1}^N d_j] ** \beta_3))

    Notice that we return the budget utilization - that is, how much of the 
    current budget should be spent servicing the current demand. This is to ensure
    that the agent doesn't overallocate.
    """

    def __init__(self, demand_expectations):
        super().__init__()

        self.b1 = nn.Parameter(torch.randn(()))
        self.b2 = nn.Parameter(torch.randn(()))
        self.b3 = nn.Parameter(torch.randn(()))

        self.demand_expectations = demand_expectations
 
    def forward(self, x: torch.tensor):
        t = x[0]
        dt = x[t + 2]

        return (dt ** self.b1 / (dt ** self.b2 + sum(self.demand_expectations[t + 1:]) ** self.b3))
    
    def string(self):
        return f'b1 = {self.b1.item()}, b2 = {self.b2.item()}, b3 = {self.b3.item()}'
    
class NNActor(nn.Module):
    """
    Three fully connected FFNN layers with ReLU activation, sigmoid at the end.

    Maps the state to action (budget utilization fraction)
    """

    def __init__(self, input_size, output_size, hidden_size1=256, hidden_size2=256, hidden_size3=256):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x: torch.tensor):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        x = self.linear4(x)
        # sigmoid to ensure that the output is in [0, 1]
        x = torch.sigmoid(x)

        return x
    
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # normally, action needs to be scaled, but since
        # our action is a fraction of the budget, we don't need to scale it
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        # enforce within [0, 1] for budget utilization
        mean = F.sigmoid(mean)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        return super(GaussianPolicy, self).to(device)
