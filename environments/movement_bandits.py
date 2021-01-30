"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class MovementBandits(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, add_action_in_obs=False):
        # new action space = [left, right]
        self.action_space = spaces.Discrete(5)

        if add_action_in_obs: 
            self.observation_space = spaces.Box(-10000000, 10000000, shape=(7,))
        else:
            self.observation_space = spaces.Box(-10000000, 10000000, shape=(6,))
        self.add_action_in_obs = add_action_in_obs

        self.realgoal = np.random.randint(0,2)

        self._seed()
        self.viewer = None
        self._reset()

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    # def randomizeCorrect(self):
    #     self.realgoal = self.np_random.randint(0,2)
        # print("new goal is " + str(self.realgoal))
    
    def draw_and_set_task(self, constraint=None, seed=None):
        """
        Draw a new task and set the environment to that task.

        Args:
            seed: Random seed that is used to generate task
            contraint:  Should be e.g. "B-Y"
        """
        if seed is None:
            seed = self.np_random.randint(9223372036854775807)

        _rnd, seed1 = seeding.np_random(seed)
        if constraint is None:
            constraint = _rnd.choice([0,1])
        # constraint should be either 0 or 1
        self.realgoal = int(constraint)

        return seed1

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print("seeded")
        return [seed]

    def _step(self, action):
        if action == 1:
            self.state[0] += 20
        if action == 2:
            self.state[0] -= 20
        if action == 3:
            self.state[1] += 20
        if action == 4:
            self.state[1] -= 20


        distance = np.mean(abs(self.state[0] - self.goals[self.realgoal][0])**2 + abs(self.state[1] - self.goals[self.realgoal][1])**2)
        # reward = -distance / 5000
        # print(distance)
        if distance < 2500:
            reward = 1
        else:
            reward = 0

        return self.obs(action), reward, False, {}

    def obs(self, action):
        obs = np.reshape(np.array([self.state] + self.goals), (-1,)) / 400
        if self.add_action_in_obs:
            obs = np.concatenate([obs, [action]])
        return obs

    def _reset(self):
        # self.randomizeCorrect()
        self.state = [200.0, 200.0]
        self.goals = []
        for x in range(2):
            self.goals.append(self.np_random.uniform(0, 400, size=(2,)))
            # self.goals.append(np.array([300, 200]))

        # self.goals.append(np.array([300, 300]))
        # self.goals.append(np.array([100, 100]))
        return self.obs(0)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            self.man_trans = rendering.Transform()
            self.man = rendering.make_circle(10)
            self.man.add_attr(self.man_trans)
            self.man.set_color(.5,.8,.5)
            self.viewer.add_geom(self.man)

            self.goal_trans = []
            for g in range(len(self.goals)):
                self.goal_trans.append(rendering.Transform())
                self.goal = rendering.make_circle(20)
                self.goal.add_attr(self.goal_trans[g])
                self.viewer.add_geom(self.goal)
                self.goal.set_color(.5,.5,g*0.8)


        self.man_trans.set_translation(self.state[0], self.state[1])
        for g in range(len(self.goals)):
            self.goal_trans[g].set_translation(self.goals[g][0], self.goals[g][1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__ == "__main__":
    env = MovementBandits()
    import ipdb; ipdb.set_trace()