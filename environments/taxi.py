#%%
import time
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
from PIL import Image
import gym

# First dimension: x, second dimension: y
# Layouts are defined by 'WALKABLE' (np array) and a map with constraints

ACTIONS = {
    0: np.array([0,0]), # No-op
    1: np.array([0,-1]),
    2: np.array([0,1]),
    3: np.array([1,0]),
    4: np.array([-1,0]),
    5: np.array([0,0]), # Pickup
    # 6: np.array([0,0]) # Dropoff
}

shape = (8,8)

TAXI_ROOMS_LAYOUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    ]).T

LOCATIONS = {
    'R': np.array([1, 1]),
    'Y': np.array([1, 6]),
    'B': np.array([5, 6]),
    'G': np.array([6, 1])
}

# test = np.zeros(shape=shape)
# test[tuple(LOCATIONS['Y'])] = 1
# print(test.T)

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class Taxi(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
        add_action_in_obs=False, 
        reward_wrong_movement=-0.1,
        image_obs=True,
        start_everywhere=False):
        self.viewer = None
        self.seed()
        self.walkable = TAXI_ROOMS_LAYOUT
        self.image_obs=image_obs
        self.start_everywhere = start_everywhere

        self.actions = ACTIONS
        if self.image_obs:
            num_layers = 3 if add_action_in_obs else 2
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.walkable.shape[0], self.walkable.shape[1], num_layers),
                dtype=np.uint8)
        else:
            # Position, passenger?, (last action)
            self.observation_space = spaces.Box(
                low=0., high=1.,
                shape=(73 if add_action_in_obs else 72,)
            )

        self.action_space = spaces.Discrete(6)
    
        self.state = {
            'loc': (0,0),
            # If passenger is in taxi, set to None
            'pas': False
        }

        self.task = {
            'pic': np.array([0,0]),
            'gol': np.array([0,0])
        }

        self.done = True

        self.draw_and_set_task()

        self.add_action_in_obs = add_action_in_obs
        self.reward_goal_found = 2
        self.reward_per_timestep = -0.1
        self.reward_wrong_movement = reward_wrong_movement
    
    def draw_and_set_task(self, constraint=None, seed=None):
        """
        Draw a new task and set the environment to that task.

        Args:
            seed: Random seed that is used to generate task
            contraint:  Should be e.g. "B-Y"
        """

        assert(constraint is None or isinstance(constraint, str))

        # Not needed here
        if seed is None:
            seed = self.np_random.randint(9223372036854775807)

        _rnd, seed1 = seeding.np_random(seed)
        if constraint is None:
            constraint = "{}-{}".format(
                _rnd.choice(['R', 'G', 'Y', 'B']),
                _rnd.choice(['R', 'G', 'Y', 'B'])
            )

        # constraint should be of the 
        constraint=constraint.split('-')
        assert len(constraint) in [2,3]

        if len(constraint) == 2 or constraint[2] == 'easy':
            self.possible_pas_states = [True, False]
        elif constraint[2] == 'hard':
            self.possible_pas_states = [False]
        else:
            raise NotImplementedError("{} no a valid modifier".format(constraint[2]))
        X, Y = np.where(self.walkable==1)
        self.possible_loc_states = [(X[i], Y[i]) for i in range(len(X))]
        if not self.start_everywhere:
            for loc in LOCATIONS.values():
                self.possible_loc_states.remove(tuple(loc))

        # Set pickup and dropoff location
        self.task['pic'] = LOCATIONS[constraint[0]]
        self.task['gol'] = LOCATIONS[constraint[1]]

        return seed1
        
    def seed(self, seed=None):
        """
        Seed must be a non-negative 
        """
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1]
    
    def reset(self):
        self.done = False
        # self.state['pas'] = np.random.choice([True, False])
        self.state['pas'] = self.np_random.choice(self.possible_pas_states)
        self.state['loc'] = self.possible_loc_states[self.np_random.choice(len(self.possible_loc_states))]

        return self.create_observation(action=0)
    
    def step(self, action):
        if self.done:
            raise Exception("Environment must be reset")

        # No-op (0) or movement (1-4)
        if action < 5: 
            mov = self.actions[action]
            loc = self.state['loc']
            new_loc = loc + mov

            if (self.check_inside_area(new_loc) and self.check_walkable(new_loc)):
                reward = self.reward_per_timestep
            else:
                reward = self.reward_wrong_movement
                new_loc = loc
            self.state['loc'] = new_loc

        # Pickup
        elif action == 5:
            if self.check_pickup_possible():
                self.state['pas'] = True
                reward = self.reward_per_timestep
            elif self.check_dropoff_possible():
                self.done = True
                reward = self.reward_goal_found
            else:
                reward = self.reward_wrong_movement

        obs = self.create_observation(action=action)
        return obs, reward, self.done, {'state': self.state, 'task': self.task}
            
    def create_observation(self, action, loc=None, layout=None):
        if loc is None:
            loc = self.state['loc']
        if layout is None:
            layout = self.walkable

        if self.image_obs:
            obs = (1 - np.copy(layout)) * 255
            obs = np.expand_dims(obs, 2)
            obs[tuple(loc)] = 125

            # All ones if picked up, otherwise all 0
            passenger_obs = np.full_like(obs, int(self.state['pas']))
            layers = [obs, passenger_obs]

            if self.add_action_in_obs:
                # Add additional channelfor last action
                layers += [np.full_like(obs, action)]
                # last_action_channel = np.full_like(obs, action)
                # obs = np.concatenate([obs, last_action_channel], axis=2)
            obs = np.concatenate(layers, axis=2)
        else: 
            obs = np.zeros(72)
            # Remove wall from state
            x = loc[0] - 1 
            y = loc[1] - 1 
            idx = self.state['pas'] * 36 + y * 6 + x
            obs[idx] = 1
            if self.add_action_in_obs:
                obs = np.concatenate([obs, np.array([action])])

        return obs

    def check_pickup_possible(self):
        return np.all(self.state['loc'] == self.task['pic']) and not self.state['pas']

    def check_dropoff_possible(self):
        return np.all(self.state['loc'] == self.task['gol']) and self.state['pas']
    
    def check_walkable(self, loc):
        return self.walkable[tuple(loc)] == 1
    
    def check_inside_area(self, loc):
        return (0 <= loc[0] < self.walkable.shape[0] and
                0 <= loc[1] < self.walkable.shape[1])

if __name__ == "__main__":
    env = Taxi(image_obs=False)
    env.reset()
    import ipdb; ipdb.set_trace()