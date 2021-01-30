"""
Adapted from:
https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
to include the possibility to set a task/goal by invoking
`env.draw_and_set_task(constraints, seed)`
"""

import numpy as np
from multiprocessing import Process, Pipe
# from . import VecEnv, CloudpickleWrapper
from baselines.common.vec_env import VecEnv, CloudpickleWrapper  
import gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def draw_and_set_task(self, constraint, seed):
    seeds = [None] * self.num_envs
    for e in range(self.num_envs):

        unwrapped_env = self.envs[e]
        while isinstance(unwrapped_env, gym.Wrapper):
            unwrapped_env = unwrapped_env.env
        seeds[e] = unwrapped_env.draw_and_set_task(constraint[e], seed[e])
    return seeds

DummyVecEnv.draw_and_set_task = draw_and_set_task

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    unwrapped_env = env
    while isinstance(unwrapped_env, gym.Wrapper):
        unwrapped_env = unwrapped_env.env
    try:
        while True:
            # print("Waiting for command...")
            # cmd, data = remote.recv()
            recieved = remote.recv()
            # print("Recieved: " + str(recieved))
            cmd, data = recieved
            # print("Command: "+cmd)
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                # print("spaces sent")
            elif cmd == 'draw_and_set_task':
                # print(data)
                seed = unwrapped_env.draw_and_set_task(data['constraint'], data['seed'])
                # print(seed)
                remote.send(seed)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class MTSubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None):
        """
        Arguments:
        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def draw_and_set_task(self, constraint, seed): 
        self._assert_not_closed()
        print(len(self.remotes), len(constraint), len(seed))
        for remote, c, s in zip(self.remotes, constraint, seed):
            data = {'constraint': c, 'seed': s}
            remote.send(('draw_and_set_task', data))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
