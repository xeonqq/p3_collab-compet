## Collaborate Compete Project Report

## Learning Algorithm

In this project, the agent's goal is to follow a moving target as long as possible. The agent has full access to the
state (33 variables) of the environment, and 4 continuous movements as its actions. Through interacting with the
environment, the agent receives positive rewards when reaches the target.

In this project, we deploy the actor-critic strategy, where we use two neural networks (actor and critic). As described
in the [ddpg paper](https://arxiv.org/abs/1509.02971). The actor network, uses the policy gradient technique to predict
the actions directly from current observed state. On the other hand, the critic network try to estimate the Q-value,
where input is the observed state and the actions, output is the Q value. The actor network is having low bias and high
variance, while the critic network is having high bias due to the dependence to temperal difference, but low variance.
Combining both networks, combines the best of two worlds, low bias and low variance.

In the project, the 20 agent unity environment is used. Detailed algorithm refer to "Algorithm 1 DDPG algorithm"
in [ddpg paper](https://arxiv.org/abs/1509.02971).

Different from the original algorithm, instead using Ornstein-Uhlenbeck process to model the noise for actions, a 0
centered Gaussian distributed noise with std-dev 0.05. The noise will decay exponentially with 0.99 every episode.

Hyperparameters:

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 5.e-4  # learning rate of the actor
LR_CRITIC = 5.e-3  # learning rate of the critic
exponential_learning_decay = 0.98  # for both actor and critic networks
```

#### Actor-Critic Network Architectures

Actor network:
![Actor network](network_diagrams/actor.png)

Critic Network:
![Critic network](network_diagrams/critic.png)

### Plot of Rewards

The training will terminate once average score reaches 30 over +100 episodes.

The network converges in **36** episodes, and kept stably at the score of ~35.

The score per episode during training:

![](ddpg_score.png)

### Demo

Runing the trained model with no exploration noise

![](result_demo.gif)

It reaches score of 38 at the end of the episode

### Progress during the project
- Version 0: Implemented the algorithm one to one from DDPG paper, 400x300 for hidden layer, same param, doesn't learn, the score is always below 1
- Version 1: Changed a different seed, remove L2 weight decay for critic network. The agent is learning, but score reach 7 then drop
sharply. See [commit](https://github.com/xeonqq/p2_continous_control/blob/51843209594a746ad9d48b38584ec8a29aece396/ddpg_score.png).
- Version 2: Since the learning is not stable, I reduced the replay buffer size, and also try to reduce learning rate to 1e-5 and 1e-4 for actor and critic network. At the same time
 Added one hidden layer to both network. And for critic network, feed action directly at the input layer together with state. Then the 
network is steadily learning till score of 14 at episode 150, but learning is too slow.
See [commit](https://github.com/xeonqq/p2_continous_control/commit/5d5c240d20db83ab6ae75ae614a704b009bd4f9c)
- Version 3 (final): increase learning rate  to 5e-5 and 5e-4 for actor and critic network, but add lr decay to ensure steady learning at the end. At the same time,
reduce the number of neurons in the hidden layers. Those changes helped a lot, which results in the final version. see [commit](https://github.com/xeonqq/p2_continous_control/commit/57086a8ae904760b4f46910f96396d09517d070a)
### Ideas for Future Work

- A huge amount of time was spent tuning hyper-parameters, a small change would result in big difference. Apply grid
  search to tune them systematically.
- Use priority reply buffer
- In the course PPO and A3C are also introduced, but for discrete action space. I wonder how they can be also used in
  continuous space.
