from model import Policy, Encoder, ValueFunction, toOnehot
from gym.spaces import Discrete
import torch.nn as nn
from distributions import FixedCategorical
import torch
import numpy as np
import copy

"""
Where are the functions in this module needed?
In main.py loop over steps:
- Sample z
- Sample a~q(a|z) + get log q(a|z) (for R-reg) + get V (for GAE)
- Evaluate p(z) and p(a|z) for R-reg

After n steps:
- (Get V for bootstrapping)
- Get U for bootstrapping

Then in the A2C module:
- Get V and and U for baseline (should be differentiable!)
- Get action_log_probs (differentiable) for loss function
"""

class HierarchicalPolicy(nn.Module):
    def __init__(self, num_tasks, num_processes_per_task, alpha,
                 obs_shape, action_space, loss, architecture, option_init):
        """
        Initialize all master policies, options and respective priors.

        Note that the value function returned by options is V(s_t,z_t) and the value function
        returned by masters is U(s_t, z_{t-1}), so we don't need to initialize them separately.


        Args:
            num_tasks (int): Number of tasks (how many different task specific policies we need)
            num_options (int): How many different options we use
            num_processes_per_task: How many environments share the same task structure?
            alpha (float) with 0<alpha<1: Prob. mass in prior master of repeating option obs_shape
            obs_shape: Observation shape of environment (already stacked if num_stack > 1)
            action_space: Action_space of environment
        """
        super(HierarchicalPolicy, self).__init__()
        self.alpha = alpha
        self.num_processes_per_task = num_processes_per_task
        self.num_tasks = num_tasks
        self.num_options = architecture['num_options']
        self.loss = loss
        self.action_space = action_space
        self.obs_shape = obs_shape
        self.architecture = architecture
        self.option_init = option_init
        # self.use_learned_master_prior = False
        # self.use_distilled_termination_prior = False

        # Initialize Master Policies
        self.master_action_space = Discrete(self.num_options)
        self.binary_action_space = Discrete(2)


        # Separate encoders for priors, masters and options
        # TODO: Make ugly workaround nicer
        # IMPORTANT: When changing stuff here, make sure the optimizer is initialized correctly!
        if architecture['shared_encoder']: 
            encoder = Encoder(obs_shape=obs_shape, architecture=architecture)
            self.encoders = nn.ModuleList([encoder, encoder, encoder])
        else:
            self.encoders = nn.ModuleList([Encoder(obs_shape=obs_shape, architecture=architecture) for i in range(3)])
        self.encoders_index = {
            "option": 0,
            "prior": 1,
            "master": 2 
        }

        self.frozen = {
            "option": False,
            "prior": False,
            "master": False
        }
        # OPTIONS
        self.options = nn.ModuleList([Policy(action_space=action_space, architecture=architecture)
                                      for j in range(num_tasks)])
        self.terminations = nn.ModuleList([Policy(action_space=self.binary_action_space, architecture=architecture)
                                      for i in range(num_tasks)])
        

        # MASTERS: During training, share encoder with options to prevent premature convergence
        self.masters = nn.ModuleList([Policy(action_space=self.master_action_space, architecture=architecture)
                                      for i in range(num_tasks)])
        self.value_functions = nn.ModuleList([ValueFunction(architecture=architecture)for j in range(num_tasks)])

        # PRIORS: Having a separate encoder allows freezing _only_ the priors will changing posteriors
        self.option_priors = Policy(action_space=action_space, architecture=architecture)
        self.termination_priors = Policy(action_space=self.binary_action_space, architecture=architecture)
        # self.master_priors = Policy(action_space=self.master_action_space, architecture=architecture)


    def split_encoder(self):
        """ Split all three encoders as they were shared before"""
        device = next(self.parameters()).device
        encoder_state_dict = self.encoders[0].state_dict()
        self.encoders = nn.ModuleList([Encoder(obs_shape=self.obs_shape, architecture=self.architecture).to(device) for i in range(3)]) 
        for i in range(3):
            self.encoders[i].load_state_dict(encoder_state_dict)


    def get_features(self, obs, z, encoder_type):
        # print("Evaluate features: {}".format(encoder_type))

        encoder = self.encoders[self.encoders_index[encoder_type]]

        num_tasks, num_processes_per_task, *obs_shape = obs.size()
        features = encoder(
            inputs=obs.contiguous().view(num_tasks * num_processes_per_task, *obs_shape),
            option=z.contiguous().view(num_tasks * num_processes_per_task, 1)
        ).view(num_tasks, num_processes_per_task, -1)

        # if not self.training and not policy_type in ["master", "value", "distilled-master"]:
        if self.frozen[encoder_type]:
            # print("Cutting gradients")
            features = features.detach()
        # else:
        #     print("Not cutting gradients")
        return features

    def reset_encoder(self, encoder_type):
        device = next(self.parameters()).device
        self.encoders[self.encoders_index[encoder_type]] = Encoder(obs_shape=self.obs_shape, architecture=self.architecture).to(device)

    def get_slice(self, task_id):
        return task_id * self.num_processes_per_task, (task_id + 1) * self.num_processes_per_task

    def reset_task_policy(self, task_id):
        flags = self.option_init['train_init_params'] if self.training else self.option_init['test_init_params']
        device = next(self.parameters()).device
        # print(flags)

        self.options[task_id] = Policy(action_space=self.action_space, architecture=self.architecture).to(device)
        self.terminations[task_id] = Policy(action_space=self.binary_action_space, architecture=self.architecture).to(device)
        self.masters[task_id] = Policy(action_space=self.master_action_space, architecture=self.architecture).to(device)
        self.value_functions[task_id] = ValueFunction(architecture=self.architecture).to(device)

        if flags['options']:
            self.options[task_id].load_state_dict(self.option_priors.state_dict())
            self.terminations[task_id].load_state_dict(self.termination_priors.state_dict())
            self.encoders[self.encoders_index['option']].load_state_dict(
                self.encoders[self.encoders_index['prior']].state_dict()
            )
        if flags['master']:
            raise NotImplementedError("Bad idea: Encoder isn't trained for value function. Create separate V encoder?")
            # self.masters[task_id].load_state_dict(copy.deepcopy(self.master_priors.state_dict()))
            self.encoders[self.encoders_index['master']].load_state_dict(
                self.encoders[self.encoders_index['prior']].state_dict()
            )

    # Sample z and a, get log q and V (for use in main.py)
    def executePolicy(self, obs, z, policy_type, b=None, masks=None, deterministic=False):
        """
        Execute policies. Either master or option as specified by type.

        batch_size = num_processes_per_task [* num_steps]

        Args:
            obs [num_tasks, batch_size]: Observations
            z [num_tasks, batch_size]: Previously (for master) or current (for policy) z
            policy_type (String): "master" or "option". Specifies which policy to execute
            masks: Mask, indicating the start of a new episode when =0

        All return values have the first two dimensions [num_tasks, batch_size]
        Return:
            value (Scalar): U(s_t,z_{t-1}) for "master" or V(s_t, z_t) for "option"
            action (Scalar for z_t, action_dimensions for a_t): z_t for "master" or a_t for "option"
            action_log_prob (Scalar): master: log q(z_t|s_t,z_{t-1}) option: log q(a_t|s_t,z_t)
        """

        assert policy_type in ["master", "option", "termination"]

        if policy_type == "termination":
            encoder_type = "option"
            policies = self.terminations
        elif policy_type == "option":
            encoder_type = "option"
            policies = self.options
        elif policy_type == "master":
            encoder_type = "master"
            policies = self.masters
            # Save old_z for b==0, i.e. when we don't change option
            old_z = z
            z = torch.full_like(z, 0)

        actions = []
        action_log_probs = []
        probs = []

        features = self.get_features(obs, z, encoder_type)
        # if not self.training and self.option_init['freeze_options_for_test'] and not policy_type == "master":
        if self.frozen[encoder_type]:
            deterministic=True

        for task_id in range(self.num_tasks):
            action, action_log_prob, prob = policies[task_id].act(features[task_id], deterministic=deterministic)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            if prob is not None:
                probs.append(prob)
            else:
                probs = None
        
        actions = torch.stack(actions, dim=0)
        action_log_probs = torch.stack(action_log_probs, dim=0)
        if probs is not None:
            probs = torch.stack(probs, dim=0)

        if policy_type == "master":
            # Only execute the master policy when we terminate an option, i.e. when b==1,
            # Otherwise just return the previous option z
            actions = actions * b + old_z * (1-b)
            # If b=0, the master must take last action with p(z_t=z_{t-1}) = 1
            b = b.float()
            action_log_probs = action_log_probs * b
            if probs is not None:
                probs = probs * b + (1-b) * toOnehot(old_z, self.num_options)
        elif policy_type == "termination":
            # At the beginning of en episode, i.e. when masks==0, return 1
            actions = actions * masks.long() + (1 - masks.long())
            # If masks=0, the termination policy must take 1 with p(b=1|mask=0) = 1
            action_log_probs = action_log_probs * masks
            b_termination_onehot = torch.zeros_like(probs)
            b_termination_onehot [:,:,1] = 1
            if probs is not None:
                probs = probs * masks + (1-masks) * b_termination_onehot

        return actions, action_log_probs, probs

    # Get differentiable log q for use in A2C
    def evaluatePolicy(self, obs, action, policy_type, z=None, b=None, masks=None):
        """
        Evaluate policies. Either master or option as specified by type.

        batch_size = num_processes_per_task [* num_steps]

        Args:
            obs [num_tasks, batch_size]: Observations on all tasks
            z [num_tasks, batch_size]: Previously (for master) or current (for policy) z
            action [num_tasks, batch_size]: Actions to evaluate
            policy_type (String): "master" or "option". Specifies which policy to execute

        All return values have the first two dimensions [num_tasks, batch_size]
        Return:
            value (Scalar): U(s_t,z_{t-1}) for "master" or V(s_t, z_t) for "option"
            action_log_prob (Scalar): master: log q(z_t|s_t,z_{t-1}) option: log q(a_t|s_t,z_t)
        """

        assert policy_type in ["termination", "master", "option"]
        if policy_type == "termination":
            encoder_type = "option"
            policies = self.terminations
        elif policy_type == "option":
            encoder_type = "option"
            policies = self.options
        elif policy_type == "master":
            encoder_type = "master"
            policies = self.masters
            # Need to use b because we don't have access to previous z
            z = torch.full_like(b, 0)
        
        # print("Evaluate Policy: {}".format(policy_type))

        dist_entropies = []
        action_log_probs = []
        features = self.get_features(obs, z, encoder_type)

        for task_id in range(self.num_tasks):
            action_log_prob, dist_entropy = policies[task_id].evaluate_actions(
                features=features[task_id], action=action[task_id]
                )
            dist_entropies.append(dist_entropy)
            action_log_probs.append(action_log_prob)

        action_log_probs = torch.stack(action_log_probs, dim=0)
        dist_entropies = torch.stack(dist_entropies, dim=0)

        # This is were it's differentiated through
        if policy_type == "master":
            # If b=0, the master must take last action with p(z_t=z_{t-1}) = 1
            # So log p = 0 and entropy = 0
            b = b.float()
            action_log_probs = action_log_probs * b
            dist_entropies = dist_entropies * b
        elif policy_type == "termination":
            # If masks=0, the termination policy must take 1 with p(b=1|mask=0) = 1
            action_log_probs = action_log_probs * masks
            dist_entropies = dist_entropies * masks

        # if not self.training and self.option_init['freeze_options_for_test'] and not policy_type == 'master':
        if self.frozen[encoder_type]:
            # print("Cutting gradients")
            action_log_probs = action_log_probs.detach()
            dist_entropies = dist_entropies.detach()
        # else:
            # print("Not cutting gradients")

        return (action_log_probs, dist_entropies)

    # Evaluate p(z) and p(a) (for use in main.py for Reg) and
    # Evaluate p(a)_\theta for training in A2C
    def evaluatePrior(self, obs, z, action, policy_type, masks=None, b=None):
        """
        Evaluate the prior probability distribution given an action.
        We don't need to split between tasks because there is only one joint prior over all tasks.

        batch_size = num_processes_per_task [* num_steps]

        Args:
            obs [num_tasks, batch_size]: Observations on all tasks
            z [num_tasks, batch_size]: Previously (for master) or current (for policy) z
            action [num_tasks, batch_size]: Action to evaluate
            policy_type (String): "master" or "option". Specifies which policy to execute

        Return:
            prior_log_prob [num_tasks, batch_size]: Prior log probability of actions
        """
        # We don't need "master" because it has a uniform prior
        assert policy_type in ["termination", "option", "distilled-termination", "distilled-master"]
        # print("Evaluate Prior: {}".format(policy_type))

        # Use the distilled termination log prob if either policy_type ==
        # "distilled-termination" (used for training) or self.use_distilled_termination_prior (used
        # for distillation during test)
        # if policy_type == "termination" and self.use_distilled_termination_prior:
        # TODO: Easier way to write this?
        if policy_type == "termination" and not self.training:
            policy_type = "distilled-termination"

        if policy_type == "termination":
            # Non-termination (i.e. 0) should have log_prob = log(alpha)
            prior_log_prob = FixedCategorical(probs=torch.tensor(
                [self.alpha, 1-self.alpha], 
                device=action.device
                )).log_probs(action)
            prior_log_prob = prior_log_prob * masks

        else: 
            if policy_type == "distilled-termination":
                priors = self.termination_priors
            if policy_type == "distilled-master":
                priors = self.master_priors
                z = torch.full_like(z, 0)
            elif policy_type == "option":
                priors = self.option_priors

            features = self.get_features(obs, z, encoder_type="prior")

            num_tasks, batch_size, *action_shape = action.size()
            num_tasks, batch_size, *features_shape = features.size()
            prior_log_prob, _ = priors.evaluate_actions(
                features=features.view(num_tasks * batch_size, *features_shape),
                action=action.contiguous().view(num_tasks * batch_size, *action_shape))
            prior_log_prob = prior_log_prob.view(num_tasks, batch_size, 1)

            if policy_type == "distilled-termination":
                prior_log_prob = prior_log_prob * masks
            if policy_type == "distilled-master":
                b = b.float()
                prior_log_prob = prior_log_prob * b
                # actions = actions * masks.long() + (1 - masks.long())

        # if not self.training and self.option_init['freeze_priors_for_test']:
        if self.frozen["prior"]:
            # print("Cutting gradients")
            prior_log_prob = prior_log_prob.detach()
        # else:
        #     print("Not cutting gradients")
        return prior_log_prob

    # Get V for baseline and to learn V
    # (used in A2C)
    def get_V(self, obs, z):
        raise NotImplementedError("Currently only wendelins loss is implemented")


    # Get U for use as baseline and bootstrap
    # Used in A2C and main.py
    def get_U(self, obs, previous_z):
        """
        Return the state-option value function U(s_{t+1}, z_t).
        They correspond to the value returned by the master policies but can also be computed as
        follows:

        U(s_{t}, z_{t-1}) = \sum_{z_t=0}^{num_options} V(s_t, z_t)
            - action_log_prob (z_t|s_t,z_{t-1}) + prior_log_prob (z_t|s_t,z_{t-1})
        
        Currently, we're computing U. 

        Args:
            obs [num_tasks, batch_size]: Observations on all tasks
            z [num_tasks, batch_size]: Previously (for master) or current (for policy) z

        Return:
            values [num_tasks, batch_size]: V(s_t,z_t)
        """

        values = []

        features = self.get_features(obs, previous_z, encoder_type="master")

        for task_id in range(self.num_tasks):
            value = self.value_functions[task_id].get_value(features[task_id])
            values.append(value)

        return torch.stack(values, dim=0)


