import numpy as np

import torch
import torch.nn as nn

# Necessary for my KFAC implementation.


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def update_current_obs(obs, current_obs, obs_shape, num_stack, num_tasks, num_processes_per_task):
    shape_dim0 = obs_shape[0]//num_stack
    obs = torch.from_numpy(obs).float()
    if num_stack > 1:
        current_obs[:, :, :-shape_dim0] = current_obs[:, :, shape_dim0:]
    # It should be ok to just reshape it because we only use obs in this function
    obs = np.reshape(obs, (num_tasks, num_processes_per_task,
                           shape_dim0, *obs_shape[1:]))
    current_obs[:, :, -shape_dim0:] = obs
    """
    # Leave this here for debugging to make sure vectorized ops do what we expect
    right_way = np.reshape(obs, (num_tasks, num_processes_per_task, 1, *obs_shape[1:]))
    wrong_way = np.zeros((num_tasks, num_processes_per_task, *obs_shape[1:]))
    for task in range(2):
        wrong_way[task] = obs[task*num_processes_per_task:(task+1)*num_processes_per_task]
    print(np.array_equal(right_way, wrong_way))
    """


def getOutputDimension(dimension, k_size, padding, stride):
    return (dimension - k_size + 2*padding) // stride + 1
