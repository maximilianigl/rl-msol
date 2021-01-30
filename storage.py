import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class RolloutStorage(object):
    def __init__(self, num_tasks, num_steps, num_processes, obs_shape, action_space, loss):
        self.num_steps = num_steps
        self.num_tasks = num_tasks
        self.num_processes_per_task = num_processes
        self.step = 0
        self.loss = loss

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        # I believe b is shifted by one?
        self.b = torch.zeros(num_tasks, num_steps, num_processes, 1).long()
        self.z = torch.zeros(num_tasks, num_steps + 1, num_processes, 1).long()
        self.obs = torch.zeros(num_tasks, num_steps + 1, num_processes, *obs_shape)
        self.actions = torch.zeros(num_tasks, num_steps, num_processes, action_shape)

        self.rewards = torch.zeros(num_tasks, num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_tasks, num_steps + 1, num_processes, 1)

        self.b_log_probs = torch.zeros(num_tasks, num_steps, num_processes, 1)
        self.b_prior_log_probs = torch.zeros(num_tasks, num_steps, num_processes, 1)
        self.z_log_probs = torch.zeros(num_tasks, num_steps, num_processes, 1)
        # self.z_prior_log_probs = torch.zeros(num_tasks, num_steps, num_processes, 1)
        self.action_log_probs = torch.zeros(num_tasks, num_steps, num_processes, 1)
        self.action_prior_log_probs = torch.zeros(num_tasks, num_steps, num_processes, 1)

        self.returns_z = torch.zeros(num_tasks, num_steps + 1, num_processes, 1)
        self.returns_a = torch.zeros(num_tasks, num_steps + 1, num_processes, 1)

        # Initialize first step of masks to 0, indicating that a new episode begins
        # This is required for correct treatment of q(z|s, z_{t-1}) and p(z|z_{t-1})
        self.masks = torch.zeros(num_tasks, num_steps + 1, num_processes, 1)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.next_value_u = None

    def cuda(self):
        self.obs = self.obs.cuda()
        self.z = self.z.cuda()
        self.b = self.b.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()

        self.b_log_probs = self.b_log_probs.cuda()
        self.b_prior_log_probs = self.b_prior_log_probs.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.action_prior_log_probs = self.action_prior_log_probs.cuda()
        self.z_log_probs = self.z_log_probs.cuda()
        # self.z_prior_log_probs = self.z_prior_log_probs.cuda()

        self.returns_z = self.returns_z.cuda()
        self.returns_a = self.returns_a.cuda()

        self.masks = self.masks.cuda()
        self.actions = self.actions.cuda()

    def insert(self, current_obs, b, z, action, value_pred,
               action_log_prob, action_prior_log_prob, z_log_prob, 
               b_log_prob, b_prior_log_prob, reward, mask):
        self.obs[:, self.step + 1].copy_(current_obs)
        self.z[:, self.step + 1].copy_(z)
        self.b[:, self.step].copy_(b)
        self.actions[:, self.step].copy_(action)

        self.value_preds[:, self.step].copy_(value_pred)

        self.b_log_probs[:, self.step].copy_(b_log_prob)
        self.b_prior_log_probs[:, self.step].copy_(b_prior_log_prob)
        self.z_log_probs[:, self.step].copy_(z_log_prob)
        # self.z_prior_log_probs[:, self.step].copy_(z_prior_log_prob)
        self.action_log_probs[:, self.step].copy_(action_log_prob)
        self.action_prior_log_probs[:, self.step].copy_(action_prior_log_prob)

        self.rewards[:, self.step].copy_(reward)
        self.masks[:, self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[:, 0].copy_(self.obs[:, -1])
        self.masks[:, 0].copy_(self.masks[:, -1])
        self.z[:, 0].copy_(self.z[:, -1])
    
    def store_next_value(self, next_value_u):
        self.next_value_u = next_value_u
    
    def get_reg_reward(self, step):
        kl_b = self.b_prior_log_probs[:,step] - self.b_log_probs[:, step] 

        if self.loss['f_div_function'] == 'tanh':
            kl_b = torch.tanh(kl_b)
        elif self.loss['f_div_function'] == 'identity':
            pass
        else: 
            raise NotImplementedError("loss['f_div_function'] '{}' not implemented".format(self.loss['f_div_function']))
        
        reg_reward = (self.rewards[:, step] * self.loss['c_r']
                        - self.z_log_probs[:, step] * (self.loss['c_ent_z'])
                        # - self.b_log_probs[:, step] * (self.loss['c_kl_b'] + self.loss['c_ent_b'])
                        - self.b_log_probs[:, step] * self.loss['c_ent_b']
                        - self.action_log_probs[:, step] * (self.loss['c_kl_a'] + self.loss['c_ent_a'])
                        # + self.b_prior_log_probs[:,step] * self.loss['c_kl_b']
                        + self.action_prior_log_probs[:, step] * self.loss['c_kl_a']
                        + kl_b * self.loss['c_kl_b'])
        return reg_reward

    def compute_returns(self):
        """
        Originally:
        - Initialize with Bootstrap value
        - R[t] = R[t+1] * gamma * mask[t+1] + r[t]

        """
        # Baseline should be the U of the _first_ z
        # next_value_u = U(s_{t+n}, z_{t+n-1})
        if self.loss['use_gae']:
            self.value_preds[:, -1] = self.next_value_u
            gae = 0
            for step in reversed(range(self.rewards.size(1))):
                reg_reward = self.get_reg_reward(step)
                delta = reg_reward + self.loss['gamma'] * self.value_preds[:, step + 1] * self.masks[:, step + 1] - self.value_preds[:, step]
                gae = delta + self.loss['gamma'] * self.loss['tau'] * self.masks[:, step + 1] * gae
                self.returns_z[:, step] = gae + self.value_preds[:, step]
        else:
            self.returns_z[:, -1] = self.next_value_u
            for step in reversed(range(self.rewards.size(1))):
                reg_reward = self.get_reg_reward(step)
                self.returns_z[:, step] = (
                    self.returns_z[:, step + 1] * self.loss['gamma'] * self.masks[:, step + 1] + reg_reward)

    def feed_forward_generator(self, 
                               old_action_log_probs,
                               old_z_log_probs,
                               old_b_log_probs,
                               values_u,
                               num_mini_batch):
        """ Used for PPO"""
        num_tasks, num_steps, num_processes = self.rewards.size()[0:3]
        batch_size = num_processes * num_steps

        # self.returns_z should have been updated
        advantages = self.returns_z[:, :-1] - values_u
        if self.loss['normalize_advt']:
            advantages = (advantages - advantages.mean()) / ( advantages.std() + 1e-5)

        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))

        mini_batch_size = batch_size // num_mini_batch
        assert mini_batch_size > 1, "Mini batch size less than or equal to 1 will completely break things"

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            # Standard stuff to return
            obs_batch = self.obs[:, :-1].view(num_tasks, -1, *self.obs.size()[3:])[:, indices]
            value_preds_batch = self.value_preds[:, :-1].view(num_tasks, -1, 1)[:, indices]
            returns_z_batch = self.returns_z[:, :-1].view(num_tasks, -1, 1)[:, indices]
            masks_batch = self.masks[:, :-1].view(num_tasks, -1, 1)[:, indices]

            # Policy choices
            actions_batch = self.actions.view(num_tasks, -1, self.actions.size(-1))[:, indices]
            z_batch_minus_one = self.z[:, :-1].view(num_tasks, -1, 1)[:, indices] 
            z_batch_plus_one = self.z[:, 1:].view(num_tasks, -1, 1)[:, indices]
            b_batch = self.b.view(num_tasks, -1, 1)[:, indices]

            # Select old policy probabilities
            # We need to use the passed in ones because the ones in storage will change:
            # They are updated to compute new returns
            old_action_log_probs_batch = old_action_log_probs.view(num_tasks, -1, 1)[:, indices]
            old_z_log_probs_batch = old_z_log_probs.view(num_tasks, -1, 1)[:, indices]
            old_b_log_probs_batch = old_b_log_probs.view(num_tasks, -1, 1)[:, indices]

            # Updated advantages
            adv_targ = advantages.view(num_tasks, -1, 1)[:, indices]

            yield (obs_batch, value_preds_batch, returns_z_batch, masks_batch, 
                   actions_batch, z_batch_minus_one, z_batch_plus_one, b_batch, 
                   old_action_log_probs_batch, old_z_log_probs_batch, old_b_log_probs_batch,
                   adv_targ)

    def update_ppo_epoch(self, hierarchical_actor_critic):
        num_tasks, num_steps, num_processes = self.rewards.size()[0:3]
        obs_shape = self.obs.size()[3:]
        action_shape = self.actions.size(-1)

        with torch.no_grad():
            b_log_probs, entropy_b = hierarchical_actor_critic.evaluatePolicy(
                obs=self.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                z=self.z[:, :-1].view(num_tasks, -1, 1),
                action=self.b.view(num_tasks, -1, 1),
                policy_type="termination",
                masks=self.masks[:, :-1].view(num_tasks, -1, 1))

            z_log_probs, entropy_z = hierarchical_actor_critic.evaluatePolicy(
                obs=self.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                b=self.b.view(num_tasks, -1, 1),
                action=self.z[:, 1:].view(num_tasks, -1, 1),
                policy_type="master")

            action_log_probs, entropy_a = hierarchical_actor_critic.evaluatePolicy(
                obs=self.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                z=self.z[:, 1:].view(num_tasks, -1, 1),
                action=self.actions.view(num_tasks, -1, action_shape),
                policy_type="option")

            # Evaluate Priors
            b_prior_log_prob = hierarchical_actor_critic.evaluatePrior(
                obs=self.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                z=self.z[:, :-1].view(num_tasks, -1, 1),
                action=self.b.view(num_tasks, -1, 1),
                policy_type="termination",
                masks=self.masks[:, :-1].view(num_tasks, -1, 1))

            action_prior_log_prob = hierarchical_actor_critic.evaluatePrior(
                obs=self.obs[:, :-1].view(num_tasks, -1, *obs_shape),
                z=self.z[:, 1:].view(num_tasks, -1, 1),
                action=self.actions.view(num_tasks, -1, action_shape),
                policy_type="option")

        # Copying to make sure dimensions are the same
        self.b_log_probs.copy_(b_log_probs.view(num_tasks, num_steps, num_processes, 1))
        self.z_log_probs.copy_(z_log_probs.view(num_tasks, num_steps, num_processes, 1))
        self.action_log_probs.copy_(action_log_probs.view(num_tasks, num_steps, num_processes, 1))
        self.b_prior_log_probs.copy_(b_prior_log_prob.view(num_tasks, num_steps, num_processes, 1))
        self.action_prior_log_probs.copy_(action_prior_log_prob.view(num_tasks, num_steps, num_processes, 1))

        self.compute_returns()
