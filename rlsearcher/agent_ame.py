import random
import logging
import ConfigSpace
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import pdb
import gym
from gym import spaces
from typing import Dict
from rlsearcher.ppo.model_ame import PolicyAME
from hpbandster.core.base_config_generator import base_config_generator

logger = logging.getLogger('AME')
logger.setLevel(logging.INFO)

class RLEnv(gym.Env):
     
    def __init__(self, configspace, num_conf, normalized_rate, device):

        self.configspace = configspace
        self.configs = dict()
        self.num_conf = num_conf
        self.normalized_rate = normalized_rate
        self.device = device
        self.ro = 1.5

        self.action_space = []
        self.observation_shape = []

        hps = self.configspace.get_hyperparameters()
        for i in range(len(hps)):
            as_shape = len(hps[i].choices)
            self.action_space.append(spaces.Discrete(as_shape))
            self.observation_shape.append(as_shape)

        self.eye_for_one_hot = [np.eye(i) for i in self.observation_shape]

        obs = int(np.sum(np.array(self.observation_shape)) + 1)
        self.observation_space = spaces.Box(low=np.float32(0.0), high=np.float32(1.0), shape=(obs, ), dtype=np.float32)

    def get_one_hot_vector(self, configs):
        state = []
        for config in configs:
            feat_conf = [self.eye_for_one_hot[idx][int(cf)] for idx, cf in enumerate(config[:-1])]
            feat_loss = [config[-1] / self.normalized_rate]
            state.append(torch.tensor(np.hstack(feat_conf + feat_loss), device=self.device).float())
        return state

    def if_not_enough_configs(self):
        budgets = list(self.configs.keys())
        if len(budgets) == 0:
            return True
        index = [len(self.configs[budget]) >= self.ro * self.num_conf for budget in budgets]
        if True in index:
            return False
        else:
            return True

    def reset(self):
        num_conf = self.num_conf

        budgets = list(self.configs.keys())
        index = [len(self.configs[budget]) >= self.ro * self.num_conf for budget in budgets]

        budgets_enough_samples = np.array(budgets)[index]
        budget = random.choice(budgets_enough_samples)
        confs = np.array(self.configs[budget])
        
        random_index = random.sample(range(0, len(confs)), num_conf)
        state = self.get_one_hot_vector(confs[random_index])
        state = [s.unsqueeze(0) for s in state]

        return state

    def step(self):
        num_conf = self.num_conf + 1

        budgets = list(self.configs.keys())
        index = [len(self.configs[budget]) >= self.ro * self.num_conf for budget in budgets]

        budgets_enough_samples = np.array(budgets)[index]
        budget = random.choice(budgets_enough_samples)
        confs = np.array(self.configs[budget])

        random_index = random.sample(range(0, len(confs)), num_conf)
        state = self.get_one_hot_vector(confs[random_index[:-1]])
        best_eval = torch.max(torch.stack(state)[:, -1])
        
        action = torch.tensor(confs[random_index[-1]][:-1], device=self.device).float()
        action_eval = torch.tensor(confs[random_index[-1]][-1], device=self.device).float()

        reward = action_eval - best_eval * self.normalized_rate
        reward = torch.clamp(torch.round(reward), min=-5., max=5.)
        state = [s.unsqueeze(0) for s in state]

        return state, action, reward


class AME(base_config_generator):
    def __init__(self, configspace, normalized_rate=100., input_num_conf=10, # random_fraction=1/3,
                 lr=0.001, max_grad_norm=0.5, cuda=True, eps=1e-5, clip_param=0.02, ppo_epoch=1,
                 num_mini_batch=5, mini_batch_size=8, value_loss_coef=0.5, entropy_coef=0.01,
                 use_clipped_value_loss=True, **kwargs):
        """
            Fits for each given budget a kernel density estimator on the best N percent of the
            evaluated configurations on this budget.

            Parameters:
            -----------
            configspace: ConfigSpace
                Configuration space object
            top_n_percent: int
                Determines the percentile of configurations that will be used as training data
                for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
                for training.
            input_num_conf: int
                minimum number of datapoints needed to fit a model

        """
        super().__init__(**kwargs)
        self.input_num_conf = input_num_conf
        self.configspace = configspace
        # self.random_fraction = random_fraction
        self.device = torch.device("cuda:0" if cuda else "cpu")
        self.epoch = 0

        self.envs = RLEnv(
            self.configspace, 
            self.input_num_conf, 
            normalized_rate,
            self.device)

        self.actor_critic = PolicyAME(
            self.envs.observation_space.shape,
            self.envs.action_space,
            self.input_num_conf)

        self.actor_critic.to(self.device)

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.conf_acc_array = -np.ones(tuple(self.envs.observation_shape))
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)


    def feed_forward_generator(self):
        batch_size = int(self.num_mini_batch * self.mini_batch_size)

        obs_all = []
        for i in range(self.input_num_conf):
            obs_all.append(torch.zeros((batch_size,) + self.envs.observation_space.shape))
        actions_all = torch.zeros(batch_size, len(self.envs.action_space))
        value_preds_all = torch.zeros(batch_size, 1)
        return_all = torch.zeros(batch_size, 1)
        old_action_log_probs_all = torch.zeros(batch_size, 1)

        for batch in range(batch_size):
            state, action, reward = self.envs.step()
            value, action_log_prob, _ = self.actor_critic.evaluate_actions(state, action.unsqueeze(1))
            for idx,s in enumerate(state):
                obs_all[idx][batch] = s
            actions_all[batch] = action
            value_preds_all[batch] = value
            return_all[batch] = reward
            old_action_log_probs_all[batch] = action_log_prob

        obs_all = [o.to(self.device) for o in obs_all]
        actions_all.to(self.device)
        value_preds_all.to(self.device)
        return_all.to(self.device)
        old_action_log_probs_all.to(self.device)

        advantages = return_all - value_preds_all
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            self.mini_batch_size,
            drop_last=True)

        for indices in sampler:
            obs_batch = [obs[indices].to(self.device) for obs in obs_all]
            actions_batch = actions_all[indices].to(self.device)
            value_preds_batch = value_preds_all[indices].to(self.device)
            return_batch = return_all[indices].to(self.device)
            old_action_log_probs_batch = old_action_log_probs_all[indices].to(self.device)
            adv_targ = advantages[indices].to(self.device)
            yield obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

    def get_config(self, budget):
        """
            Function to sample a new configuration

            This function is called inside Hyperband to query a new configuration


            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled

            returns: config
                should return a valid configuration

        """
        
        logger.debug('start sampling a new configuration.')
        

        sample = None
        info_dict = {}
        
        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        # if np.random.rand() < self.random_fraction or self.envs.if_not_enough_configs():
        if self.envs.if_not_enough_configs():
            sample =  self.configspace.sample_configuration().get_dictionary()

            conf = ConfigSpace.Configuration(self.envs.configspace, sample)
            while self.conf_acc_array[tuple(np.int32(conf.get_array()))] >= 0:
                sample =  self.configspace.sample_configuration().get_dictionary()
                conf = ConfigSpace.Configuration(self.envs.configspace, sample)
            self.conf_acc_array[tuple(np.int32(conf.get_array()))] = 0

            info_dict['model_based_pick'] = False

        else:
            logger.debug('start sampling a new configuration by RL.')
            state = self.envs.reset()
            with torch.no_grad():
                action = self.actor_critic.act(state)
            sample = ConfigSpace.Configuration(self.configspace, vector=action[0]).get_dictionary()
            info_dict['model_based_pick'] = True

            conf = ConfigSpace.Configuration(self.envs.configspace, sample)
            while self.conf_acc_array[tuple(np.int32(conf.get_array()))] >= 0:
                sample =  self.configspace.sample_configuration().get_dictionary()
                conf = ConfigSpace.Configuration(self.envs.configspace, sample)
                info_dict['model_based_pick'] = False
            self.conf_acc_array[tuple(np.int32(conf.get_array()))] = 0

        print('config: %s, model_based_pick: %r' % (sample, info_dict['model_based_pick']))

        return sample, info_dict

    def new_result(self, job, update_model=True):
        """
            function to register finished runs

            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.


            Parameters:
            -----------
            job: hpbandster.distributed.dispatcher.Job object
                contains all the info about the run
        """

        super().new_result(job)

        if job.result is None:
            # One could skip crashed results, but we decided to assign a zero loss
            acc = 0
        else:
            # same for non numeric losses.
            acc = job.result["loss"] if np.isfinite(job.result["loss"]) else 0

        budget = job.kwargs["budget"]

        if budget not in self.envs.configs.keys():
            self.envs.configs[budget] = []

        # We want to get a numerical representation of the configuration in the original space
        conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
        self.envs.configs[budget].append(np.append(conf.get_array(), [acc]))
        self.conf_acc_array[tuple(np.int32(conf.get_array()))] = acc
        
        # skip model training:
        if (not update_model) or self.envs.if_not_enough_configs():
            return
        
        num_updates = self.num_mini_batch

        for e in range(self.ppo_epoch):

            data_generator = self.feed_forward_generator()

            loss_epoch = 0
            value_loss_epoch = 0
            action_loss_epoch = 0

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                                    old_action_log_probs_batch, adv_targ = sample
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch, True)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                loss_epoch += loss.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()

            loss_epoch /= num_updates
            value_loss_epoch /= num_updates
            action_loss_epoch /= num_updates
                
            # logger.info('Training RL Agent (AME) - Epoch: [%d], Loss: %f, Value_Loss: %f, Action_Loss: %f'\
            #        %(e, loss_epoch, value_loss_epoch, action_loss_epoch))
            print('Training RL Agent (AME) - Epoch: [%d], Loss: %f, Value_Loss: %f, Action_Loss: %f'\
                   %(self.epoch, loss_epoch, value_loss_epoch, action_loss_epoch))
            self.epoch += 1