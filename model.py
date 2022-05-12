import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from device import device


def layer_init(layer, bound=None):
    # nn.init.orthogonal_(layer.weight.data)
    # layer.weight.data.mul_(w_scale)
    if bound is None:
        bound = 1 / np.sqrt(layer.in_features)
    nn.init.uniform_(layer.bias.data, -bound, bound)
    nn.init.uniform_(layer.weight.data, -bound, bound)
    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), seed=0, bn=True, gate=F.relu):
        super(FCBody, self).__init__()
        self.seed = torch.manual_seed(seed)
        dims = (state_dim,) + hidden_units
        self._bn=bn
        self.bn = nn.BatchNorm1d(state_dim)
        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]
        self.in_features = state_dim

    def forward(self, x):
        if self._bn:
            x = self.bn(x)
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
                 state_body,
                 critic_body,
                 seed,
                 ):
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_body = state_body
        self.critic_body = critic_body
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 3e-4)

        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters()) + list(
            self.state_body.parameters())

        self.to(device)

    def forward(self, obs, action):
        xs = self.state_body(obs)
        xs = torch.cat((xs, action), dim=1)
        xs = self.critic_body(xs)
        value = self.fc_critic(xs)
        return value
