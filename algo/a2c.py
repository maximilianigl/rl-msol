import torch
import torch.nn as nn
import torch.optim as optim

class A2C(object):
    def __init__(self,
                 hierarchical_actor_critic,
                 loss,
                 opt):

        self.hierarchical_actor_critic = hierarchical_actor_critic
        self.max_grad_norm = opt['max_grad_norm']
        self.loss = loss
        self.opt = opt
        # self.update_priors = True
        self.init_optimizer(agent=hierarchical_actor_critic)

    def init_optimizer(self, agent):
        # params = agent.parameters()
        master_lr = self.opt['lr']
        if agent.training:
            master_lr *= self.opt['master_lr_factor']
        
        params = []
        master_params = []
        for name, param in agent.named_parameters():
            if name.startswith("masters") or name.startswith("encoders.2"):
                master_params.append(param)
            else:
                params.append(param)

        self.optimizer = optim.RMSprop([{'params': params, 'lr': self.opt['lr']},
                                        {'params': master_params, 'lr': master_lr}], eps=self.opt['eps'], alpha=self.opt['alpha'])


    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[3:]
        action_shape = rollouts.actions.size()[-1]
        num_tasks, num_steps, num_processes_per_task, _ = rollouts.rewards.size()

        """
        Get differentiable baseline values (which will also be trained).
        Get differentiable log probs.

        In rollouts:
        - Compute n-step targets

        We need:
        - Loss for option prior
        - Loss for V
        - Loss for Master
        - Loss for Option
        """

        # Recompute all the log probs that we wanna differentiate through
        # Output dimensions [num_tasks, num_tasks * num_steps, 1]

        # Evaluate Policies

        ############## Evaluate Priors

        if self.loss['b_distillation'] == 'master':
            with torch.no_grad():
                # Draw z_master with all b=1, i.e. let it choose freshly
                z_master, _, _ = self.hierarchical_actor_critic.executePolicy(
                    obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                    z=rollouts.z[:, :-1].view(num_tasks, -1, 1), # should be ignored
                    b=torch.ones_like(rollouts.b.view(num_tasks, -1, 1)),
                    policy_type="master",
                )

            delta_b = 1 - (z_master == rollouts.z[:, :-1].view(num_tasks, -1, 1))

            b_prior_log_prob = self.hierarchical_actor_critic.evaluatePrior(
                obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                z=rollouts.z[:, :-1].view(num_tasks, -1, 1),
                # action=rollouts.b.view(num_tasks, -1, 1),
                action=delta_b,
                policy_type="distilled-termination",
                masks=rollouts.masks[:, :-1].view(num_tasks, -1, 1))
        elif self.loss['b_distillation'] == 'posterior':
            b_prior_log_prob = self.hierarchical_actor_critic.evaluatePrior(
                obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                z=rollouts.z[:, :-1].view(num_tasks, -1, 1),
                action=rollouts.b.view(num_tasks, -1, 1),
                policy_type="distilled-termination",
                masks=rollouts.masks[:, :-1].view(num_tasks, -1, 1))
        else:
            raise NotImplementedError("b_distillation type {} not implemented".format(self.loss['b_distillation']))


        # z_prior_log_prob = self.hierarchical_actor_critic.evaluatePrior(
        #     obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
        #     z=rollouts.z[:, :-1].view(num_tasks, -1, 1),
        #     b=rollouts.b.view(num_tasks, -1, 1),
        #     action=rollouts.z[:, 1:].view(num_tasks, -1, 1),
        #     policy_type="distilled-master")

        action_prior_log_prob = self.hierarchical_actor_critic.evaluatePrior(
            obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
            z=rollouts.z[:, 1:].view(num_tasks, -1, 1),
            action=rollouts.actions.view(num_tasks, -1, action_shape),
            policy_type="option")

        ################# Evaluate Posteriors
        
        b_log_probs, entropy_b = self.hierarchical_actor_critic.evaluatePolicy(
            obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
            z=rollouts.z[:, :-1].view(num_tasks, -1, 1),
            action=rollouts.b.view(num_tasks, -1, 1),
            policy_type="termination",
            masks=rollouts.masks[:, :-1].view(num_tasks, -1, 1))

        z_log_probs, entropy_z = self.hierarchical_actor_critic.evaluatePolicy(
            obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
            b=rollouts.b.view(num_tasks, -1, 1),
            action=rollouts.z[:, 1:].view(num_tasks, -1, 1),
            policy_type="master")

        action_log_probs, entropy_a = self.hierarchical_actor_critic.evaluatePolicy(
            obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
            z=rollouts.z[:, 1:].view(num_tasks, -1, 1),
            action=rollouts.actions.view(num_tasks, -1, action_shape),
            policy_type="option")

        values_u = self.hierarchical_actor_critic.get_U(
            obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
            previous_z=rollouts.z[:, :-1].view(num_tasks, -1, 1))


        # TODO: Write down math and check again!
        z_log_probs = z_log_probs.view(num_tasks, num_steps, num_processes_per_task, 1)
        action_log_probs = action_log_probs.view(num_tasks, num_steps, num_processes_per_task, 1)
        b_log_probs = b_log_probs.view(num_tasks, num_steps, num_processes_per_task, 1)
        action_prior_log_prob = action_prior_log_prob.view(num_tasks, num_steps, num_processes_per_task, 1)
        b_prior_log_prob = b_prior_log_prob.view(num_tasks, num_steps, num_processes_per_task, 1)
        # z_prior_log_prob = z_prior_log_prob.view(num_tasks, num_steps, num_processes_per_task, 1)
        # values_v = values_v.view(num_tasks, num_steps, num_processes_per_task, 1)
        values_u = values_u.view(num_tasks, num_steps, num_processes_per_task, 1)

        # Currently only wendelin's loss is implemented
        advantages_z = rollouts.returns_z[:, :-1] - values_u

        # So far we only learn V
        value_loss = advantages_z.pow(2).mean()
        action_loss_a = -(advantages_z.detach() * action_log_probs).mean() 
        action_loss_z = -(advantages_z.detach() * z_log_probs).mean()
        action_loss_b = -(advantages_z.detach() * b_log_probs).mean()

        # TODO: What happens with gamma here (or in general for A2C)? Is the sign correct?
        action_prior_loss = - action_prior_log_prob.mean()
        b_prior_loss = - b_prior_log_prob.mean()
        # z_prior_loss = - z_prior_log_prob.mean()

        self.optimizer.zero_grad()

        # update_master = (j % self.loss['master_update_freq'] == 0) or (not
        # self.hierarchical_actor_critic.training)
        
        elc_a = self.loss['elc_a'] if self.loss['elc_a'] is not None else self.loss['elc']
        elc_b = self.loss['elc_b'] if self.loss['elc_b'] is not None else self.loss['elc']
        elc_z = self.loss['elc_z'] if self.loss['elc_z'] is not None else self.loss['elc']

        # Whether or not gradients are cut is determined in the hierarchical_actor_critic
        (value_loss * self.loss['value_loss_coef']
         + action_loss_a * self.loss['action_loss_coef_a']
         + action_loss_z * self.loss['action_loss_coef_z'] 
         + action_loss_b * self.loss['action_loss_coef_b']
         + action_prior_loss * self.loss['prior_loss_coef'] 
         + b_prior_loss * self.loss['prior_loss_coef'] 
        #  + z_prior_loss * self.loss['prior_loss_coef'] 
        - entropy_a.mean() * elc_a 
        - entropy_b.mean() * elc_b 
        - entropy_z.mean() * elc_z).backward()
        #  - entropy_a.mean() * self.loss['entropy_loss_coef']
        #  - entropy_b.mean() * self.loss['entropy_loss_coef']
        #  - entropy_z.mean() * self.loss['entropy_loss_coef']).backward()

        nn.utils.clip_grad_norm_(self.hierarchical_actor_critic.parameters(),
                                 self.max_grad_norm)

        ####################### Debugging ###########################
        # # Checking whether all parameters are updated....
        # old_params = {}
        # for name, param in self.hierarchical_actor_critic.named_parameters():
        #     old_params[name] = param.clone()
        ####################### End ###########################

        self.optimizer.step()

        ####################### Debugging ###########################
        # # ...continued: Checking whether all parameters are updated
        # for name, param in self.hierarchical_actor_critic.named_parameters():
        #     if torch.equal(old_params[name], param):
        #         print(name)
        ####################### End ###########################
        losses = {
            'value_loss': value_loss.item(),
            'action_loss_a': action_loss_a.item(),
            'action_loss_z': action_loss_z.item(),
            'action_loss_b': action_loss_b.item(),
            'action_prior_loss': action_prior_loss.item(),
            'b_prior_loss': b_prior_loss.item(),
            'entropy_a': entropy_a.mean().item(),
            'entropy_b': entropy_b.mean().item(),
            'entropy_z': entropy_z.mean().item()
        }

        return losses
