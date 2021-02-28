from sacred.observers import FileStorageObserver
import copy
import glob
import os
import time

import gym
import environments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from subproc_multitask_vec_env import MTSubprocVecEnv, DummyVecEnv
from multitask_vec_normalize import MTVecNormalize
from envs import make_env, WrapPyTorch
# from model import Policy
from hierarchical_policy import HierarchicalPolicy
from storage import RolloutStorage
from utils import update_current_obs, getOutputDimension, update_linear_schedule

import algo

# For sacred
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver
import logging
from torch.utils import data
from itertools import islice

from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Create Experiment
ex = Experiment("distral")

# Add configuration
ex.add_config('./default.yaml')

# First try to connect to given db. If that doesn't work, save results to FileStorageObserver
maxSevSelDelay = 20  # Assume 1ms maximum server selection delay
# Check whether server is accessible
print("ONLY FILE STORAGE OBSERVER ADDED")
ex.observers.append(FileStorageObserver.create('saved_runs'))

LONG_NUMBER = 2147483647


def printHeader():
    logging.info(
        '      Progr | FPS | avg | med | min | max  ')
    logging.info(
        '      ------|-----|-----|-----|-----|----- ')


@ex.config
def configuration(environment, architecture, tasks, test_tasks, num_steps, num_test_steps, loss):

    if test_tasks is None:
        test_tasks = tasks

    if num_test_steps is None:
        num_test_steps = num_steps

    if loss['c_kl_a'] is None:
        loss['c_kl_a'] = float(loss['c_kl_b'])
        loss['fixed_a'] = False
    else:
        loss['fixed_a'] = True
    loss['c_kl_b_orig'] = loss['c_kl_b']
    loss['c_kl_a_orig'] = loss['c_kl_a']

    if loss['entropy_loss_coef_test'] is None:
        loss['entropy_loss_coef_test'] = loss['entropy_loss_coef_0']


@ex.capture
def reset_task(restart_tasks, hierarchical_actor_critic, constraint, agent, returned_task_seed, envs, testing_envs,
               tasks, num_processes):
    num_tasks = len(tasks)
    num_processes_per_task = num_processes // num_tasks

    for next_restart_task in restart_tasks:
        print("Resetting task {} ({})".format(next_restart_task,
                                              constraint[next_restart_task * num_processes_per_task]))
        hierarchical_actor_critic.reset_task_policy(task_id=next_restart_task)

        # Set a new task seed for all processes of the task that needs to be replaced
        # Constraints remain the same
        new_task_seed = np.random.randint(LONG_NUMBER)
        low, high = hierarchical_actor_critic.get_slice(next_restart_task)
        for i in range(low, high):
            returned_task_seed[i] = new_task_seed

        returned_task_seed = envs.draw_and_set_task(
            constraint=constraint,
            seed=returned_task_seed)
        testing_envs.draw_and_set_task(
            constraint=constraint,
            seed=returned_task_seed)
    agent.init_optimizer(hierarchical_actor_critic)

    return returned_task_seed


@ex.capture
def save_model(model, name, envs, save_dir, _run):
    name_model = os.path.join(save_dir, name)
    torch.save(model.state_dict(), name_model)
    s_current = os.path.getsize(name_model) / (1024 * 1024)
    logging.info('Saving model {}: Size: {} MB'.format(name, s_current))
    _run.add_artifact(name_model)
    os.remove(name_model)

    # Saving the observation normalization
    if isinstance(envs, MTVecNormalize) and envs.ob_rms is not None:
        np_name = os.path.join(save_dir, '{}.npy'.format(name))
        np.save(np_name, {
            'mean': envs.ob_rms.mean,
            'var': envs.ob_rms.var,
            'count': envs.ob_rms.count
        })
        s_current = os.path.getsize(np_name) / (1024 * 1024)
        logging.info('Saving ob_rms {}: Size: {} MB'.format(
            np_name, s_current))
        _run.add_artifact(np_name)
        os.remove(np_name)


@ex.capture
def test_policy(testing_envs, hierarchical_actor_critic,
                tasks, num_steps, num_processes, num_stack, loss, cuda):
    num_tasks = len(tasks)
    num_processes_per_task = num_processes // num_tasks
    obs_shape = testing_envs.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
    device = next(hierarchical_actor_critic.parameters()).device

    if testing_envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = testing_envs.action_space.shape[0]

    current_obs = torch.zeros(
        num_tasks, num_processes_per_task, *obs_shape).to(device)

    masks = torch.zeros(num_tasks, num_processes_per_task, 1).to(device)
    z = torch.zeros(num_tasks, num_processes_per_task, 1).long().to(device)

    obs = testing_envs.reset()
    update_current_obs(obs, current_obs, obs_shape, num_stack,
                       num_tasks, num_processes_per_task)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_tasks, num_processes_per_task, 1])
    final_rewards = torch.zeros([num_tasks, num_processes_per_task, 1])

    if cuda:
        current_obs = current_obs.cuda()

    finished_environments = torch.zeros((num_processes,))

    while (finished_environments == 0).any():

        with torch.no_grad():
            b, b_log_prob, _ = hierarchical_actor_critic.executePolicy(
                obs=current_obs,
                z=z,
                policy_type="termination",
                masks=masks,
                deterministic=True
            )

            z, z_log_prob, _ = hierarchical_actor_critic.executePolicy(
                obs=current_obs,
                z=z,
                policy_type="master",
                b=b,
                deterministic=True
            )
            action, action_log_prob, _ = hierarchical_actor_critic.executePolicy(
                obs=current_obs,
                z=z,
                policy_type="option",
                deterministic=True
            )

        _, _, *action_shape = action.size()
        flat_action = action.view(
            num_tasks * num_processes_per_task, *action_shape)
        cpu_actions = flat_action.squeeze(1).cpu().numpy()

        # Obser reward and next obs
        obs, reward, done, info = testing_envs.step(cpu_actions)

        single_obs_shape = obs.shape[1:]
        obs = np.reshape(
            np.stack(obs), (num_tasks, num_processes_per_task) + single_obs_shape)
        reward = np.reshape(
            np.stack(reward), (num_tasks, num_processes_per_task))
        done = np.reshape(np.stack(done), (num_tasks, num_processes_per_task))

        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 2)).float()

        episode_rewards += reward

        # If done then clean the history of observations.
        masks = torch.ones(
            (num_tasks, num_processes_per_task, 1), dtype=torch.float32)
        for task in range(num_tasks):
            for process in range(num_processes_per_task):
                masks[task, process] = 0.0 if done[task][process] else 1.0

        # Mask rewards
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        # Mask observations
        finished_environments += (1-masks).view(-1)
        masks = masks.to(device)
        if current_obs.dim() == 5:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks

        update_current_obs(obs, current_obs, obs_shape,
                           num_stack, num_tasks, num_processes_per_task)

    return final_rewards


@ex.automain
def main(algorithm, opt, loss, ppo, normalization,
         alpha, seed, num_processes, num_steps, num_test_steps,
         num_stack, log_interval, test_log_interval,
         num_frames, reset_encoder_in_test, freeze_in_test,
         environment, tasks, test_tasks, architecture, num_env_restarts,
         warmup_period_frames, final_period_frames, load_id,
         testing_frames, option_init, num_simultaneous_restarts,
         save_dir, cuda, add_timestep, _run):

    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ACKTR currently broken
    assert algorithm in ['a2c', 'ppo']

    # If all tasks are ints, convert them to actual ints
    try:
        tasks = list(map(int, tasks))
        test_tasks = list(map(int, test_tasks))
    except:
        pass

    num_tasks = len(tasks)
    num_processes_per_task = num_processes // num_tasks
    # num_frames = num_frames PER TASK
    num_updates = int(num_frames) * num_tasks // num_steps // num_processes
    print('Num updates:{}\n'.format(num_updates))
    assert num_updates > 0, 'num_updates is 0, increase number of frames'

    # There will be `num_env_restarts` within the time between warmup_updates:(num_updates -
    # final_updates)
    # This leaves some warmup period and final training period to inspect the fully trained options
    warmup_updates = int(warmup_period_frames) * \
        num_tasks // num_steps // num_processes
    final_updates = int(final_period_frames) * \
        num_tasks // num_steps // num_processes
    testing_updates = int(testing_frames) * \
        num_tasks // num_test_steps // num_processes

    restart_interval = (num_updates - warmup_updates -
                        final_updates) // (num_env_restarts + 1)

    print('Num tasks:{}\nNum processes per task:{}\n'.format(
        num_tasks, num_processes_per_task))

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    print("#######")
    print(
        "WARNING: All rewards are clipped or normalized, but we are plotting the average return after clipping. Sacred plots will be inaccurate if per-timestep rewards are out of the range [-1, 1]")
    print("#######")

    torch.set_num_threads(1)

    envs = [make_env(environment, seed, i, add_timestep)
            for i in range(num_tasks * num_processes_per_task)]
    testing_envs = [make_env(environment, seed, i, add_timestep)
                    for i in range(num_tasks * num_processes_per_task)]
    constraint = []
    test_constraint = []
    task_seed = []
    for task in tasks:
        constraint += [task]*num_processes_per_task
        task_seed += [np.random.randint(LONG_NUMBER)]*num_processes_per_task
    for task in test_tasks:
        test_constraint += [task]*num_processes_per_task

    if num_processes > 1:
        envs = MTSubprocVecEnv(envs)
        testing_envs = MTSubprocVecEnv(testing_envs)
    else:
        envs = DummyVecEnv(envs)
        testing_envs = DummyVecEnv(testing_envs)

    if len(envs.observation_space.shape) == 1:
        envs = MTVecNormalize(
            envs, ob=normalization['ob'], ret=normalization['ret'], gamma=loss['gamma'])
        testing_envs = MTVecNormalize(
            testing_envs, ob=normalization['ob'], ret=False, gamma=loss['gamma'])

    returned_task_seed = envs.draw_and_set_task(
        constraint=constraint,
        seed=task_seed)
    testing_envs.draw_and_set_task(
        constraint=constraint,
        seed=returned_task_seed
    )

    print("Task seeds: {}".format(returned_task_seed))

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])

    hierarchical_actor_critic = HierarchicalPolicy(
        num_tasks, num_processes_per_task, alpha,
        obs_shape, envs.action_space, loss, architecture,
        option_init=option_init)

    if load_id is not None:
        docs = get_docs(db_uri, db_name, 'runs')
        doc = docs.find_one({'_id': load_id})
        name = "model_after_training"
        # config = doc['config']
        # config.update({'num_processes': len(config['tasks']), 'cuda': False})
        file_id = get_file_id(doc=doc, file_name=name)
        save_file_from_db(file_id=file_id, destination='model_tmp_{}.pyt'.format(
            _run._id), db_uri=db_uri, db_name=db_name)
        state_dict = torch.load("model_tmp_{}.pyt".format(
            _run._id), map_location=lambda storage, loc: storage)
        hierarchical_actor_critic.load_state_dict(state_dict)
        os.remove('model_tmp_{}.pyt'.format(_run._id))
        print("Loading model parameters complete.")

        if isinstance(envs, MTVecNormalize) and envs.ob_rms is not None:
            print("Loading ob_rms normalization")
            ob_name = name + ".npy"
            file_id = get_file_id(doc=doc, file_name=ob_name)
            save_file_from_db(
                file_id=file_id, destination='ob_rms_tmp.npy', db_uri=db_uri, db_name=db_name)
            rms_dict = np.load("ob_rms_tmp.npy")[()]
            print(rms_dict)
            envs.ob_rms.mean = rms_dict['mean']
            envs.ob_rms.var = rms_dict['var']
            envs.ob_rms.count = rms_dict['count']
            testing_envs.ob_rms.mean = rms_dict['mean']
            testing_envs.ob_rms.var = rms_dict['var']
            testing_envs.ob_rms.count = rms_dict['count']
            os.remove("ob_rms_tmp.npy")

    num_parameters = 0
    for p in hierarchical_actor_critic.parameters():
        num_parameters += p.nelement()

    num_params_master = 0
    for p in hierarchical_actor_critic.masters[0].parameters():
        num_params_master += p.nelement()

    num_params_option = 0
    for p in hierarchical_actor_critic.options[0].parameters():
        num_params_option += p.nelement()

    print(hierarchical_actor_critic)
    print("Total Number parameters: {}".format(num_parameters))
    print("Number parameters master: {}".format(num_params_master))
    print("Number parameters option: {}".format(num_params_option))

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if cuda:
        hierarchical_actor_critic.cuda()

    if algorithm == 'a2c':
        agent = algo.A2C(hierarchical_actor_critic, loss=loss, opt=opt)
    elif algorithm == 'ppo':
        agent = algo.PPO(hierarchical_actor_critic, loss, opt, ppo)
    elif algorithm == 'acktr':
        raise NotImplementedError("ACKTR not implemented with HRL")
        # agent = algo.A2C_ACKTR(hierarchical_actor_critic, value_loss_coef,
        #                        entropy_coef, acktr=True)

    def reset_envs(storage_length):
        rollouts = RolloutStorage(num_tasks, storage_length, num_processes_per_task,
                                  obs_shape, envs.action_space, loss)
        current_obs = torch.zeros(
            num_tasks, num_processes_per_task, *obs_shape)

        obs = envs.reset()

        update_current_obs(obs, current_obs, obs_shape,
                           num_stack, num_tasks, num_processes_per_task)
        for task in range(num_tasks):
            rollouts.obs[task, 0].copy_(current_obs[task])
        if cuda:
            current_obs = current_obs.cuda()
            rollouts.cuda()

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([num_tasks, num_processes_per_task, 1])
        final_rewards = torch.zeros([num_tasks, num_processes_per_task, 1])
        episode_length = torch.zeros([num_tasks, num_processes_per_task, 1])
        final_length = torch.zeros([num_tasks, num_processes_per_task, 1])
        episode_terminations = torch.zeros(
            [num_tasks, num_processes_per_task, 1])
        final_terminations = torch.zeros(
            [num_tasks, num_processes_per_task, 1])
        master_terminations = torch.zeros(
            [num_tasks, num_processes_per_task, 1])
        final_master_terminations = torch.zeros(
            [num_tasks, num_processes_per_task, 1])
        return (rollouts, current_obs, episode_rewards, final_rewards,
                episode_length, final_length, episode_terminations, final_terminations,
                master_terminations, final_master_terminations)

    rollouts, current_obs, episode_rewards, final_rewards, episode_length, final_length, \
        episode_terminations, final_terminations, master_terminations, final_master_terminations = reset_envs(
            storage_length=num_steps)

    start = time.time()
    hierarchical_actor_critic.train()
    rollout_length = num_steps
    assert num_tasks >= num_simultaneous_restarts
    randomSampler = data.sampler.BatchSampler(data.sampler.RandomSampler(range(num_tasks)),
                                              batch_size=num_simultaneous_restarts,
                                              drop_last=True)
    rndSampler_iter = iter(randomSampler)
    iterator = iter(range(num_updates + testing_updates))

    for j in iterator:

        # Load old model if load_id is given
        if load_id is not None and j == 0:
            # Skip to j == num_updates - 1
            next(islice(iterator, num_updates-2, num_updates-2), None)
            j = next(iterator)
            ppo['use_linear_clip_decay'] = False
            opt['use_lr_decay'] = False

        # Updated Learning rate
        j_mod = j % num_updates
        lr_schedule_length = num_updates if j <= num_updates else testing_updates
        if opt['use_lr_decay']:
            update_linear_schedule(
                agent.optimizer, j_mod, lr_schedule_length, opt['lr'])

        # Update clip param
        if algorithm == 'ppo' and ppo['use_linear_clip_decay']:
            agent.clip_param = ppo['clip_param'] * \
                (1 - j_mod / float(lr_schedule_length))

        # Update c_kl_b
        if loss['c_kl_b_1'] is not None:
            per = np.clip((j-warmup_updates)/(num_updates-final_updates), 0, 1)
            cur_val = (1-per) * loss['c_kl_b_orig'] + per * loss['c_kl_b_1']
            rollouts.loss['c_kl_b'] = cur_val
            if not loss['fixed_a']:
                rollouts.loss['c_kl_a'] = cur_val

        # Update c_kl_a
        if loss['c_kl_a_1'] is not None:
            per = np.clip((j-warmup_updates)/(num_updates-final_updates), 0, 1)
            cur_val = (1-per) * loss['c_kl_a_orig'] + per * loss['c_kl_a_1']
            rollouts.loss['c_kl_a'] = cur_val
            # if not loss['fixed_b']:
            #     rollouts.loss['c_kl_a'] = cur_val

        # Update entropy_coef
        train_progress = j/(num_updates-final_updates)
        if not agent.hierarchical_actor_critic.training:
            # Testing
            elc = loss['entropy_loss_coef_test']
        elif loss['entropy_loss_coef_1'] is not None:
            factor = max(0, 1-train_progress)
            elc = (loss['entropy_loss_coef_0'] * factor +
                   loss['entropy_loss_coef_1'] * (1 - factor))
        else:
            elc = loss['entropy_loss_coef_0']
        loss['elc'] = elc

        for step in range(rollout_length):
            # Sample actions
            """
            Note regarding z:
            z_t is treated the same way as s_t with regards to saving because at t=0 we need access to
            s_{-1} and z_{t-1}. HOWEVER, that means that the code is off by one compared to the
            equations:
            In equations: z_t depends on s_t and z_{t-1}
            Here: z_t depends on s_{t-1} and z_{t-1}
            """

            with torch.no_grad():
                b, b_log_prob, _ = hierarchical_actor_critic.executePolicy(
                    obs=rollouts.obs[:, step],
                    z=rollouts.z[:, step],
                    policy_type="termination",
                    masks=rollouts.masks[:, step]
                )

                z, z_log_prob, _ = hierarchical_actor_critic.executePolicy(
                    obs=rollouts.obs[:, step],
                    z=rollouts.z[:, step],
                    policy_type="master",
                    b=b
                )

                action, action_log_prob, _ = hierarchical_actor_critic.executePolicy(
                    obs=rollouts.obs[:, step],
                    z=z,
                    policy_type="option"
                )

                # Evaluate Log probs for regularized reward
                b_prior_log_prob = hierarchical_actor_critic.evaluatePrior(
                    obs=rollouts.obs[:, step],
                    z=rollouts.z[:, step],
                    action=b,
                    policy_type="termination",
                    masks=rollouts.masks[:, step]
                )
                action_prior_log_prob = hierarchical_actor_critic.evaluatePrior(
                    obs=rollouts.obs[:, step],
                    z=z,
                    action=action,
                    policy_type="option"
                )
                value_pred = hierarchical_actor_critic.get_U(
                    obs=rollouts.obs[:, step],
                    previous_z=z
                )

            # Flatten actions:
            _, _, *action_shape = action.size()
            flat_action = action.view(
                num_tasks * num_processes_per_task, *action_shape)
            cpu_actions = flat_action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            single_obs_shape = obs.shape[1:]
            obs = np.reshape(
                np.stack(obs), (num_tasks, num_processes_per_task) + single_obs_shape)
            reward = np.reshape(
                np.stack(reward), (num_tasks, num_processes_per_task))
            done = np.reshape(
                np.stack(done), (num_tasks, num_processes_per_task))

            reward = torch.from_numpy(
                np.expand_dims(np.stack(reward), 2)).float()

            episode_rewards += reward
            episode_length += 1
            episode_terminations += b.cpu().float()

            delta_b = 1 - (z == rollouts.z[:, step]).int()
            master_terminations += delta_b.cpu().float()

            # If done then clean the history of observations.
            masks = torch.ones(
                (num_tasks, num_processes_per_task, 1), dtype=torch.float32)
            for task in range(num_tasks):
                for process in range(num_processes_per_task):
                    masks[task, process] = 0.0 if done[task][process] else 1.0

            # Mask rewards
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            final_length *= masks
            final_length += (1 - masks) * episode_length
            episode_length *= masks

            final_terminations *= masks
            # It starts of with a termination
            final_terminations += (1 - masks) * (episode_terminations - 1)
            episode_terminations *= masks

            final_master_terminations *= masks
            # It starts of with a termination
            final_master_terminations += (1 - masks) * \
                (master_terminations - 1)
            master_terminations *= masks

            # Mask observations
            if cuda:
                masks = masks.cuda()
            if current_obs.dim() == 5:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs, obs_shape,
                               num_stack, num_tasks, num_processes_per_task)

            rollouts.insert(
                current_obs=current_obs,
                z=z,
                b=b,
                action=action,
                value_pred=value_pred,
                action_log_prob=action_log_prob,
                action_prior_log_prob=action_prior_log_prob,
                z_log_prob=z_log_prob,
                b_log_prob=b_log_prob,
                b_prior_log_prob=b_prior_log_prob,
                reward=reward,
                mask=masks)

        with torch.no_grad():
            # obs[-1] is s_{t+1} in equations
            # z[-1] is z_{t} in equations
            # Basically: Those are the last values we know which are s_{t+1} and z_t
            next_value_u = hierarchical_actor_critic.get_U(
                obs=rollouts.obs[:, -1],
                previous_z=rollouts.z[:, -1])

        rollouts.store_next_value(next_value_u)
        rollouts.compute_returns()
        losses = agent.update(rollouts)

        rollouts.after_update()

        # While still in training and in between warmup_updates and final_updates
        if warmup_updates < j < num_updates and j < (num_updates - final_updates) and (j - warmup_updates) % restart_interval == 0:
            # Get tasks to reset
            try:
                next_restart_tasks = next(rndSampler_iter)
            except StopIteration as e:
                rndSampler_iter = iter(randomSampler)
                next_restart_tasks = next(rndSampler_iter)

            returned_task_seed = reset_task(
                next_restart_tasks, hierarchical_actor_critic, constraint, agent,
                returned_task_seed, envs, testing_envs)
            # load_master=train_load_master_params)

            # Unfortunately there isn't a simple nice way to only restart the environment that was resetted
            rollouts, current_obs, episode_rewards, final_rewards, episode_length, final_length,\
                episode_terminations, final_terminations, master_terminations, final_master_terminations = reset_envs(
                    storage_length=num_steps)

        # When we reached the end of the training phase, reset all tasks
        if j == num_updates - 1:
            save_model(hierarchical_actor_critic, "model_after_training", envs)
            print("Reset all tasks, stop updating prior, start testing")
            last_training_task_seed = returned_task_seed.copy()
            hierarchical_actor_critic.eval()
            returned_task_seed = reset_task(
                restart_tasks=range(num_tasks),
                hierarchical_actor_critic=hierarchical_actor_critic,
                constraint=test_constraint,
                agent=agent,
                returned_task_seed=returned_task_seed,
                envs=envs,
                testing_envs=testing_envs)

            print("Freezing and resetting for test")
            hierarchical_actor_critic.frozen['prior'] = freeze_in_test['prior']
            hierarchical_actor_critic.frozen['option'] = freeze_in_test['option']

            if architecture['shared_encoder']:
                hierarchical_actor_critic.split_encoder()

            # This will create a new Encoder!
            if reset_encoder_in_test['option']:
                hierarchical_actor_critic.reset_encoder('option')

            if reset_encoder_in_test['master']:
                hierarchical_actor_critic.reset_encoder('master')
            agent.init_optimizer(hierarchical_actor_critic)

            # Unfortunately there isn't a simple nice way to only restart the environment that was resetted
            rollouts, current_obs, episode_rewards, final_rewards, episode_length, final_length,\
                episode_terminations, final_terminations, master_terminations, final_master_terminations = reset_envs(
                    storage_length=num_test_steps)
            rollout_length = num_test_steps

        if (j < num_updates and j % log_interval == 0) or (j >= num_updates and j % test_log_interval == 0):

            test_performance = test_policy(
                testing_envs, hierarchical_actor_critic)
            end = time.time()
            if j % (log_interval*10) == 0:
                printHeader()

            if j < num_updates:
                total_num_steps = (j + 1) * num_processes * num_steps
            else:
                total_num_steps = (
                    num_updates * num_steps + (j + 1 - num_updates) * num_test_steps) * num_processes

            # FPS PER TASK (because num_frames is also per task!)
            fps = int(total_num_steps / num_tasks / (end - start))

            logging.info('Updt: {:5} |{:5} {:5}|{:5}|{:5}|{:5}'.format(
                str(j / num_updates)[:5],
                str(fps),
                str(final_rewards.mean().item())[:5],
                str(final_rewards.median().item())[:5],
                str(final_rewards.min().item())[:5],
                str(final_rewards.max().item())[:5],
            ))

            for task in range(num_tasks):
                _run.log_scalar('return.avg.{}'.format(task), float(
                    final_rewards[task].mean()), total_num_steps//num_tasks)
                _run.log_scalar('return.test.avg.{}'.format(task), float(
                    test_performance[task].mean()), total_num_steps//num_tasks)

            _run.log_scalar('return.avg', final_rewards.mean(
            ).item(), total_num_steps//num_tasks)
            _run.log_scalar('return.test.avg', test_performance.mean(
            ).item(), total_num_steps//num_tasks)
            _run.log_scalar('episode.length', final_length.mean(
            ).item(), total_num_steps//num_tasks)
            _run.log_scalar('episode.terminations', final_terminations.mean(
            ).item(), total_num_steps//num_tasks)
            _run.log_scalar('episode.master_terminations', final_master_terminations.mean(
            ).item(), total_num_steps//num_tasks)
            _run.log_scalar('fps', fps, total_num_steps//num_tasks)

            _run.log_scalar(
                'loss.value', losses['value_loss'], total_num_steps//num_tasks)
            _run.log_scalar(
                'loss.action_a', losses['action_loss_a'], total_num_steps//num_tasks)
            _run.log_scalar(
                'loss.action_z', losses['action_loss_z'], total_num_steps//num_tasks)
            _run.log_scalar(
                'loss.action_b', losses['action_loss_b'], total_num_steps//num_tasks)
            _run.log_scalar(
                'loss.action_prior', losses['action_prior_loss'], total_num_steps//num_tasks)
            _run.log_scalar(
                'loss.b_prior', losses['b_prior_loss'], total_num_steps//num_tasks)
            _run.log_scalar('loss.entropy_a',
                            losses['entropy_a'], total_num_steps//num_tasks)
            _run.log_scalar('loss.entropy_b',
                            losses['entropy_b'], total_num_steps//num_tasks)
            _run.log_scalar('loss.entropy_z',
                            losses['entropy_z'], total_num_steps//num_tasks)

    _run.info["seeds_final"] = returned_task_seed
    # _run.info["last_training_task_seed"] = last_training_task_seed
    _run.info["constraints_final"] = constraint
    _run.info['test_constraints_final'] = test_constraint

    save_model(hierarchical_actor_critic, "final_model", envs)
