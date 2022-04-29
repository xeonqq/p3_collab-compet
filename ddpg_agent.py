import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from device import device
from model import ActorNet, CriticNet, FCBody
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 5.e-4  # learning rate of the actor
LR_CRITIC = 5.e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay


class DDPG_Agent(object):
    def __init__(self, state_dim, action_dim, seed=101, update_interval=1, n_updates=1):
        print("Using device: ", device)
        # self._actor_local = ActorNet(state_dim, action_dim, seed).to(device)
        # self._actor_target = ActorNet(state_dim, action_dim, seed).to(device)
        self.seed = seed
        np.random.seed(seed)
        self._actor_local = ActorNet(action_dim, FCBody(state_dim, (128, 200, 128), seed), seed)
        self._actor_target = ActorNet(action_dim, FCBody(state_dim, (128, 200, 128), seed), seed)
        self._actor_optimizer = optim.Adam(self._actor_local.parameters(), lr=LR_ACTOR)

        # self._critic_local = CriticNet(state_dim, action_dim, seed).to(device)
        # self._critic_target = CriticNet(state_dim, action_dim, seed).to(device)
        self._critic_local = CriticNet(FCBody(state_dim + action_dim, (200, 400), seed), FCBody(400, (200,), seed),
                                       seed)
        self._critic_target = CriticNet(FCBody(state_dim + action_dim, (200, 400), seed), FCBody(400, (200,), seed),
                                        seed)
        self._critic_optimizer = optim.Adam(self._critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self._replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self._ou_noise = OUNoise(action_dim, seed)
        self._noise_decay = 0.99

        self.soft_update(self._actor_local, self._actor_target, 1)
        self.soft_update(self._critic_local, self._critic_target, 1)
        self._t = 0
        self._update_interval = update_interval
        self._n_updates = n_updates

        # summary(self._actor_local, (state_dim,))

        print(self._actor_local)
        print(self._critic_local)

    def get_critic_optimizer(self):
        return self._critic_optimizer

    def get_actor_optimizer(self):
        return self._actor_optimizer

    def load_actor_model(self, model):
        self._actor_local.load_state_dict(torch.load(model))

    def step_with_replay_buffer(self, states, actions, rewards, next_states, dones):
        self._replay_buffer.add(states, actions, rewards, next_states, dones)
        # Learn, if enough samples are available in memory
        if len(self._replay_buffer) > BATCH_SIZE:
            # if self._t % self._update_interval == 0:
            #     for _ in range(self._n_updates):
            experiences = self._replay_buffer.sample()
            self.learn(experiences)

    def step(self, states, actions, rewards, next_states, dones):
        self._t += 1
        self.step_with_replay_buffer(states, actions, rewards, next_states, dones)

    def reset(self):
        self._ou_noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self._actor_target(next_states)
        # target_return = rewards + GAMMA * self._critic_target(next_states, next_actions).detach()
        target_next_Q = self._critic_target(next_states, next_actions)
        target_return = rewards + GAMMA * target_next_Q * (1 - dones)
        expected_return = self._critic_local(states, actions)

        critic_loss = F.mse_loss(target_return, expected_return)
        # print(critic_loss)
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._critic_local.parameters(), 10)
        self._critic_optimizer.step()

        expected_action = self._actor_local(states)
        # self._critic_local.eval()
        # with torch.no_grad():
        actor_loss = -self._critic_local(states, expected_action).mean()
        # print(actor_loss)
        # self._critic_local.train()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor_local.parameters(), 15)
        self._actor_optimizer.step()

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

    def act(self, states, noise, n_episode):
        states = torch.from_numpy(states).float().to(device)
        self._actor_local.eval()
        with torch.no_grad():
            actions = self._actor_local(states)
        self._actor_local.train()
        actions = actions.cpu().data.numpy()
        if noise:
            # noise = self._ou_noise.sample()
            noise = np.random.normal(0, 0.05, np.shape(actions))
            noise_decay = self._noise_decay ** (5 * n_episode)
            noise *= noise_decay
            actions += noise
        return np.clip(actions, -1, 1)  # all actions between -1 and 1
