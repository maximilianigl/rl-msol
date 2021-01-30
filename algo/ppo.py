import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 hierarchical_actor_critic,
                 loss,
                 opt,
                 ppo):

        self.hierarchical_actor_critic = hierarchical_actor_critic
        self.loss = loss
        self.opt = opt
        self.ppo = ppo
        self.update_priors = True
        self.clip_param = self.ppo['clip_param']
        self.clipped_value_loss = self.ppo['clip_value_loss']
        self.init_optimizer(agent=hierarchical_actor_critic)

    def init_optimizer(self, agent):
        params = agent.parameters()
        self.optimizer = optim.Adam(params, lr=self.opt['lr'], eps=self.opt['eps'])
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

    def update(self, rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        action_prior_loss_epoch = 0

        b_prior_loss_epoch = 0
        entropy_a_epoch = 0
        entropy_z_epoch = 0
        entropy_b_epoch = 0

        num_tasks, num_steps, num_processes, *obs_shape = rollouts.obs.size()

        ### Things that shoudn't change during ppo_epochs (instead of advantage and value):
        # V_{\pi-old}
        # [num_tasks, num_steps + 1, num_processes_per_task, 1]
        values_u  = rollouts.value_preds[:, :-1].clone()

        # \pi-old (We need cloen b)
        old_action_log_probs = rollouts.action_log_probs.clone()
        old_z_log_probs = rollouts.z_log_probs.clone()
        old_b_log_probs = rollouts.b_log_probs.clone()
        # And rewards but they are saved in rollouts
        # What we need to update is the log-fractions appearing in the advantage

        for e in range(self.ppo['ppo_epoch']):
            # with torch.no_grad():
            #     values_u = self.hierarchical_actor_critic.get_U(
            #         obs=rollouts.obs[:, :-1].view(num_tasks, -1, *obs_shape),
            #         previous_z = rollouts.z[:, :-1].view(num_tasks, -1, 1)).view(num_tasks, num_steps - 1, num_processes, 1)

            #advantages_z = rollouts.returns_z[:, :-1] - rollouts.value_preds[:, :-1]

            data_generator = rollouts.feed_forward_generator(
                old_action_log_probs=old_action_log_probs,
                old_z_log_probs=old_z_log_probs,
                old_b_log_probs=old_b_log_probs,
                values_u=values_u,
                num_mini_batch=self.ppo['num_mini_batch'])
            
            for sample in data_generator:
                (obs_batch, value_preds_batch, returns_z_batch, masks_batch,
                    actions_batch, z_batch_minus_one, z_batch_plus_one, b_batch,
                    old_action_log_probs_batch, old_z_log_probs_batch, old_b_log_probs_batch,
                    adv_targ) = sample

                #####################################################################
                # Evaluate Policies
                b_log_probs, entropy_b = self.hierarchical_actor_critic.evaluatePolicy(
                    obs=obs_batch,
                    z=z_batch_minus_one,
                    action=b_batch,
                    policy_type="termination",
                    masks=masks_batch)

                z_log_probs, entropy_z = self.hierarchical_actor_critic.evaluatePolicy(
                    obs=obs_batch,
                    b=b_batch,
                    action=z_batch_plus_one,
                    policy_type="master")

                action_log_probs, entropy_a = self.hierarchical_actor_critic.evaluatePolicy(
                    obs=obs_batch,
                    z=z_batch_plus_one,
                    action=actions_batch,
                    policy_type="option")

                # Evaluate Priors
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

                action_prior_log_prob = self.hierarchical_actor_critic.evaluatePrior(
                    obs=obs_batch,
                    z=z_batch_plus_one,
                    action=actions_batch,
                    policy_type="option")

                values_u_batch = self.hierarchical_actor_critic.get_U(
                    obs=obs_batch,
                    previous_z=z_batch_minus_one)
                #####################################################################
                ratio_a = torch.exp(action_log_probs - old_action_log_probs_batch)
                ratio_z = torch.exp(z_log_probs - old_z_log_probs_batch)
                ratio_b = torch.exp(b_log_probs - old_b_log_probs_batch)

                def mclamp(value):
                    return torch.clamp(value,
                                       1.0 - self.clip_param,
                                       1.0 + self.clip_param)

                # joint_ratio = torch.exp(num - denom)
                joint_surr_1 = ratio_a * ratio_z * ratio_b * adv_targ.detach()
                if self.ppo['ppo_loss_type'] == 'joint':

                    joint_surr_2 = mclamp(ratio_a * ratio_z * ratio_b) * adv_targ.detach()

                    action_loss = -torch.min(joint_surr_1, joint_surr_2).mean()

                elif self.ppo['ppo_loss_type'] == 'individual':

                    indv_surr_2 = mclamp(ratio_a) * mclamp(ratio_z) * mclamp(ratio_b) * adv_targ.detach()
                    
                    action_loss = -torch.min(joint_surr_1, indv_surr_2).mean()

                else:
                    raise NotImplementedError("ppo_loss_type {} is not supported".format(self.ppo['ppo_loss_type']))

                if self.clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values_u_batch - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values_u_batch - returns_z_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - returns_z_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = F.mse_loss(returns_z_batch, values_u_batch)

                action_prior_loss = - action_prior_log_prob.mean()
                b_prior_loss = - b_prior_log_prob.mean()

                self.optimizer.zero_grad()

                elc_a = self.loss['elc_a'] if self.loss['elc_a'] is not None else self.loss['elc']
                elc_b = self.loss['elc_b'] if self.loss['elc_b'] is not None else self.loss['elc']
                elc_z = self.loss['elc_z'] if self.loss['elc_z'] is not None else self.loss['elc']

                (value_loss * self.loss['value_loss_coef']
                 + action_loss * self.loss['action_loss_coef_a']
                 + action_prior_loss * self.loss['prior_loss_coef'] * self.update_priors # Yes, multiplying with booleans does what you'd expect
                 + b_prior_loss * self.loss['prior_loss_coef'] * self.update_priors # Yes, multiplying with booleans does what you'd expect
                 - entropy_a.mean() * elc_a 
                 - entropy_b.mean() * elc_b 
                 - entropy_z.mean() * elc_z).backward()

                nn.utils.clip_grad_norm_(self.hierarchical_actor_critic.parameters(),
                                         self.opt['max_grad_norm'])
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                action_prior_loss_epoch += action_prior_loss.item()
                b_prior_loss_epoch += b_prior_loss.item()
                entropy_a_epoch += entropy_a.mean().item()
                entropy_z_epoch += entropy_z.mean().item()
                entropy_b_epoch += entropy_b.mean().item()

            rollouts.update_ppo_epoch(self.hierarchical_actor_critic)

        #self.scheduler.step()

        num_updates = self.ppo['ppo_epoch'] * self.ppo['num_mini_batch']

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        action_prior_loss_epoch /= num_updates
        b_prior_loss_epoch /= num_updates
        entropy_a_epoch /= num_updates
        entropy_z_epoch /= num_updates
        entropy_b_epoch /= num_updates

        losses = {
            'value_loss': value_loss_epoch,
            'action_loss_a': action_loss_epoch,
            'action_loss_z': action_loss_epoch,
            'action_loss_b': action_loss_epoch,
            'action_prior_loss': action_prior_loss_epoch,
            'b_prior_loss': b_prior_loss_epoch,
            'entropy_a': entropy_a_epoch,
            'entropy_z': entropy_z_epoch,
            'entropy_b': entropy_b_epoch
        }

        return losses
