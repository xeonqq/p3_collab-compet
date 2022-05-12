import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from device import device
from model import ActorNet, CriticNet, FCBody
from ou_noise import OUNoise
from torch.optim.lr_scheduler import ExponentialLR

GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 3.e-3  # learning rate of the actor
LR_CRITIC = 3.e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay


class DDPG_Agent(object):
    def __init__(self, state_dim, action_dim, agent_id, n_agents, seed, update_interval=1, n_updates=1):
        print("Using device: ", device)
        self.seed = seed
        self._agent_id = agent_id
        self._n_agents = n_agents
        np.random.seed(seed)
        hidden_units = (64,64)
        self._actor_local = ActorNet(action_dim, FCBody(state_dim, hidden_units, seed), seed)
        self._actor_target = ActorNet(action_dim, FCBody(state_dim, hidden_units, seed), seed)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=LR_ACTOR)

        state_feature_out = 64
        hidden_units = (64,)
        self._critic_local = CriticNet(FCBody(state_dim * self._n_agents, (state_feature_out,), seed),
                                       FCBody(state_feature_out + action_dim * self._n_agents, hidden_units, seed),
                                       seed)
        self._critic_target = CriticNet(FCBody(state_dim * self._n_agents, (state_feature_out,), seed),
                                        FCBody(state_feature_out + action_dim * self._n_agents, hidden_units, seed),
                                        seed)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self._ou_noise = OUNoise(action_dim, seed)
        self._noise_decay_base = 0.98
        self._noise_decay = 1

        self.soft_update(self._actor_local, self._actor_target, 1)
        self.soft_update(self._critic_local, self._critic_target, 1)

        lr_decay = 0.999
        self._actor_scheduler = ExponentialLR(self._actor_optimizer, gamma=lr_decay)
        self._critic_scheduler = ExponentialLR(self._critic_optimizer, gamma=lr_decay)

        print(self._actor_local)
        print(self._critic_local)

    @property
    def agent_id(self):
        return self._agent_id

    def get_critic_optimizer(self):
        return self._critic_optimizer

    def get_actor_optimizer(self):
        return self._actor_optimizer

    def load_actor_model(self, model):
        self._actor_local.load_state_dict(torch.load(model))

    def reset_noise(self, n_episode):
        self._ou_noise.reset()
        self._noise_decay = self._noise_decay_base ** (n_episode / 6)

    def step_scheduler(self, n_episode):
        if n_episode%20==0:
            self._actor_scheduler.step()
            self._critic_scheduler.step()

    def critic_learn(self, obs, actions, next_actions, next_obs, rewards, dones):
        # target_return = rewards + GAMMA * self._critic_target(next_states, next_actions).detach()
        with torch.no_grad():
            target_next_Q = self._critic_target(next_obs, next_actions)
            target_return = rewards + GAMMA * target_next_Q * (
                    1 - dones[:, self._agent_id])
        expected_return = self._critic_local(obs, actions)

        critic_loss = F.mse_loss(target_return, expected_return)

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

    def actor_learn(self, obs, local_actions):
        actor_loss = -self._critic_local(obs, local_actions).mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor_local.parameters(), 0.1)
        self._actor_optimizer.step()

        # al = actor_loss.cpu().detach().item()
        # cl = critic_loss.cpu().detach().item()
        # print('agent%i/losses' % self._agent_id,
        #                    {'critic loss': cl,
        #                     'actor_loss': al})

    def soft_update_target_weights(self):
        self.soft_update(self._actor_local, self._actor_target, TAU)
        self.soft_update(self._critic_local, self._critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def target_act(self, states):
        actions = self._actor_target(states)
        return actions

    def local_act(self, states):
        actions = self._actor_local(states)
        return actions

    def act(self, states, n_episode, noise_stddev=0.05):
        states = states[np.newaxis,...]
        states = torch.from_numpy(states).float().to(device)
        self._actor_local.eval()
        with torch.no_grad():
            actions = self._actor_local(states)
        self._actor_local.train()
        actions = actions.cpu().data.numpy()
        if noise_stddev > 0:
            noise = self._ou_noise.sample()
            # noise = np.random.normal(0, noise_stddev, np.shape(actions))
            # if (n_episode%100==0):
            #     print(n_episode, self._noise_decay)
            noise *= self._noise_decay
            # if n_episode%100==0:
            #     print("episode: {} decay: {}, noise: {}".format(n_episode, noise_decay, noise))
            actions += noise
        return np.clip(actions, -1, 1)  # all actions between -1 and 1
