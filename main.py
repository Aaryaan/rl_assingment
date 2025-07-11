import torch as to
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from collections import deque, namedtuple
import random

def revenue_response_dynamics(day, trend, periodicity, baseline, p, s, elasticities):
    # revenue dynamics toy - elasticities
    # first order price curve, no higher order effects of price
    return baseline *(1 + trend * day /365 + np.log1p(elasticities[0]*p) + np.log1p(elasticities[1]*s) +
                      0.3*(np.sin(2 * np.pi * day/365 * periodicity)+ 1))

class BudgetEnv:
    """
    Episode = one campaign/month of H days.
    Observation contains for each SKU its baseline demand and date,
    plus the remaining episode budget.
    Action = continuous vector of length 2*Nb for SKU share and then promo vs search"""

    def __init__(self, num_skus = 50, horizon = 14, episode_budget = 100, seed =  29):
        self.num_skus = num_skus
        self.horizon = horizon
        self.init_budget = episode_budget
        self.rng = np.random.default_rng(seed)
        self.ob_dim = num_skus + 2                # n of state vars: baseline + remBudget + day
        self.act_dim = 2 * num_skus               # hierarchical action
        self.reset()

    def _baseline(self):
        # constant baseline demand profile
        return self.rng.uniform(3, 10, size=self.num_skus)

    def _periodicity(self):
        # SKU seasonality
        return self.rng.integers(1, 13, size=self.num_skus)

    def reset(self):
        self.day = 0
        self.remaining_budget = self.init_budget
        # per episode latent SKU parameters
        self.baseline = self._baseline()
        self.periodicity = self._periodicity()
        self.trend = self.rng.uniform(-0.05, 0.2)
        self.search_el = self.rng.normal(0.02, 0.005, size=self.num_skus)
        self.promo_el = self.rng.normal(0.03, 0.008, size=self.num_skus)
        return self._obs()

    def step(self, action):

        # split & transform
        sku_logits, promo_logits = np.split(action, 2)
        w = np.exp(sku_logits) / np.exp(sku_logits).sum()          # softmax
        rho = 1.0 / (1.0 + np.exp(-promo_logits))                  # sigmoid

        # Static allocation demo. Can change to dynamic allocation with lagrange multiplier budget constraint.
        todays_budget = self.remaining_budget / (self.horizon - self.day)
        spend_each = w * todays_budget
        promo_spend = spend_each * rho
        search_spend = spend_each - promo_spend

        # rewards from dynamics
        revenue = revenue_response_dynamics(self.day, self.trend, self.periodicity, self.baseline, promo_spend, search_spend,
                                                [self.promo_el, self.search_el])
        reward = revenue.sum().astype(float)

        # logging and progressing
        self.remaining_budget -= spend_each.sum()
        self.day += 1
        done = (self.day >= self.horizon) or (self.remaining_budget <= 0) # For lambda extension

        info = dict(revenue=float(revenue.sum()), spend=float(spend_each.sum()), remaining=float(self.remaining_budget))
        return self._obs(), reward, done, info

    def _obs(self):
        # concat state variables
        rem = np.full((1,), self.remaining_budget / self.init_budget)
        day = np.full((1,), self.day)

        return np.concatenate([self.baseline, rem, day]).astype(np.float32)


# Replay Buffer


class ReplayBuffer:
    def __init__(self, capacity = 100000):
        self.buf = deque(maxlen=capacity)
        self.Transition = namedtuple("Transition",
                        field_names=("state", "action", "reward",
                                     "next_state", "done"))

    def add(self, *args):
        self.buf.append(self.Transition(*[np.asarray(a) for a in args]))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        batch = self.Transition(*zip(*batch))
        # convert to tensors
        device = to.device("cuda" if to.cuda.is_available() else "cpu")
        return tuple(to.as_tensor(np.stack(x), dtype=to.float32,device=device) for x in batch)

    def __len__(self):
        return len(self.buf)


#NN Archetecture
def mlp(in_dim, out_dim, hidden=(128, 128), act=nn.ReLU):
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    """ Policy Approximator"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.backbone = mlp(obs_dim, act_dim)

    def forward(self, obs):
        return self.backbone(obs)

class Critic(nn.Module):
    """ Q approximator"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, obs, act):
        x = to.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)


# Model
class DDPGAgent:
    def __init__(self,
                    obs_dim,
                    act_dim,
                    device = None,
                    actor_lr=1e-4,
                    critic_lr=5e-4,
                    gamma=0.99,
                    tau=0.005):

        self.device = (device if device is not None
                       else to.device("cuda" if to.cuda.is_available()
                                         else "cpu"))
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.target_actor = Actor(obs_dim, act_dim).to(self.device)
        self.target_critic = Critic(obs_dim, act_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.act_optim = to.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.crit_optim = to.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

    def soft_update(self): # Target param update
        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)


    def select_action(self, obs, noise_std=0.1):
        obs = to.as_tensor(obs, dtype=to.float32, device=self.device).unsqueeze(0)
        with to.no_grad():
            a = self.actor(obs).cpu().numpy()[0]
        a += noise_std * np.random.randn(*a.shape)
        return np.clip(a, -10.0, 10.0).astype(np.float32)

    def update(self, batch, gamma):
        obs, act, rew, next_obs, done = batch
        # Critic loss
        with to.no_grad():
            next_act = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, next_act)
            y = rew + gamma * (1 - done) * target_q
        q = self.critic(obs, act)
        loss_q = to.nn.functional.mse_loss(q, y)
        self.crit_optim.zero_grad()
        loss_q.backward()
        self.crit_optim.step()

        # Actor loss  (L = -Q )
        pred_act = self.actor(obs)
        actor_loss = -self.critic(obs, pred_act).mean()

        self.act_optim.zero_grad()
        actor_loss.backward()
        self.act_optim.step()

        # target networks
        self.soft_update()

# Training
def train(num_episodes = 500,
          batch_size = 128):
    env = BudgetEnv(episode_budget = 200)
    agent = DDPGAgent(obs_dim=env.ob_dim, act_dim=env.act_dim)
    buffer = ReplayBuffer()

    returns = []
    for ep in range(num_episodes):
        obs = env.reset()
        done, ep_ret = False, 0.0
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_ret += reward

            if len(buffer) >= 500:             # warm up buffer before learning
                batch = buffer.sample(batch_size)
                agent.update(batch, gamma=agent.gamma)

        returns.append(ep_ret)
        if (ep + 1) % 20 == 0:
            avg = np.mean(returns[-20:])
            print(f"Episode {ep+1:<4d}| Avg return = {avg:8.1f}")

    print("Training complete.")
    return agent

# Train cycle
train(1000,32)


1 == 1