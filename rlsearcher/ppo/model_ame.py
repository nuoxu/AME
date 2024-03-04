import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from rlsearcher.ppo.distributions import Bernoulli, Categorical, DiagGaussian
from rlsearcher.ppo.utils import init
from rlsearcher.ppo.GTrXL import StableTransformerXL

class PolicyAME(nn.Module):
    def __init__(self, obs_shape, action_space, num_conf, hidden_size=64):
        super(PolicyAME, self).__init__()

        self.base = MLPBase(obs_shape, hidden_size*num_conf, hidden_size)

        self.dist = nn.ModuleList()
        for asp in action_space:
            if asp.__class__.__name__ == "Discrete":
                self.dist.append(Categorical(self.base.output_size, asp.n))
            else:
                raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError
    
    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist_feature = [d(actor_features) for d in self.dist]

        if deterministic:
            act = [df.mode().cpu().numpy() for df in dist_feature]
        else:
            act = [df.sample().cpu().numpy() for df in dist_feature]

        return np.hstack(act)

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action, multi_batch=False):
        value, actor_features = self.base(inputs, multi_batch)
        dist_feature = [d(actor_features) for d in self.dist]

        action_log_probs = []
        dist_entropy = []

        if multi_batch:
            for idx, d in enumerate(dist_feature):
                action_log_probs.append(d.log_probs(action[:,idx].unsqueeze(1)))
                dist_entropy.append(d.entropy().mean())
        else:
            for idx, d in enumerate(dist_feature):
                action_log_probs.append(d.log_probs(action[idx]))
                dist_entropy.append(d.entropy().mean())

        action_log_probs = torch.cat(action_log_probs, 1).mean(1).unsqueeze(1)
        dist_entropy = torch.stack(dist_entropy).mean()

        return value, action_log_probs, dist_entropy


class MLPBase(nn.Module):
    def __init__(self, obs_shape, num_inputs, hidden_size, n_transformer_layers=3, n_attn_heads=4):
        super(MLPBase, self).__init__()

        self._hidden_size = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.input_feat = nn.Sequential(
            init_(nn.Linear(obs_shape[0], hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.transformer = StableTransformerXL(d_input=hidden_size, n_layers=n_transformer_layers, 
            n_heads=n_attn_heads, d_head_inner=32, d_ff_inner=64)
        self.memory = None
        self.memory2 = None

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs, multi_batch=False):

        x = [self.input_feat(inp) for inp in inputs]
        # x = torch.cat(x, -1)
        x = torch.stack(x)

        if multi_batch:
            tx = self.transformer(x, self.memory)
            x, self.memory = tx["logits"], tx['memory']
        else:
            tx = self.transformer(x, self.memory2)
            x, self.memory2 = tx["logits"], tx['memory']

        x = x.view((x.shape[1],-1))

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor
