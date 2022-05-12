import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from device import device


def layer_init(layer, bound=None):
    # nn.init.orthogonal_(layer.weight.data)
    # layer.weight.data.mul_(w_scale)
    if bound is None:
        bound = 1/np.sqrt(layer.in_features)
    nn.init.uniform_(layer.bias.data, -bound, bound)
    nn.init.uniform_(layer.weight.data, -bound, bound)
    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), seed=0, gate=F.relu):
        super(FCBody, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims = (state_dim,) + hidden_units
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class ActorNet(nn.Module):
    def __init__(self,
                 action_dim,
                 phi_body,
                 seed
                 ):
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.phi_body = phi_body
        self.fc_action = layer_init(nn.Linear(phi_body.feature_dim, action_dim), 3e-3)

        self.actor_params = list(self.fc_action.parameters())
        self.phi_params = list(self.phi_body.parameters())

        self.to(device)

    def forward(self, obs):
        phi = self.phi_body(obs)

        action = self.fc_action(phi)
        return torch.tanh(action)


class CriticNet(nn.Module):
    def __init__(self,
                 critic_body,
                 seed,
                 ):
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.critic_body = critic_body
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 3e-4)

        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())

        self.to(device)

    def forward(self, obs, action):
        xs = torch.cat((obs, action), dim=1)
        xs = self.critic_body(xs)
        value = self.fc_critic(xs)
        return value
