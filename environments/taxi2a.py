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

MOVE = {
    0: np.array([0, 1]), # No-op
    1: np.array([0,-1]), # Turn
    2: np.array([1 ,0]),
    3: np.array([-1,0]),
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

class Taxi2A(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
        add_action_in_obs=False, 
        reward_wrong_movement=-0.1,
        image_obs=True,
        start_everywhere=False,
        scramble_prob=0.1):
        self.viewer = None
        self.seed()
        self.walkable = TAXI_ROOMS_LAYOUT
        self.image_obs=image_obs
        self.start_everywhere = start_everywhere
        self.scramble_prob = scramble_prob

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
                shape=(289 if add_action_in_obs else 288,)
            )

        self.action_space = spaces.Discrete(5)
    
        self.state = {
            'loc': (0,0),
            # If passenger is in taxi, set to None
            'pas': False,
            'rot': 0
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
        self.state['rot'] = self.np_random.choice([0,1,2,3])

        return self.create_observation(action=0)
    
    def step(self, action):
        if self.done:
            raise Exception("Environment must be reset")

        # No-op (0) or movement (1-4)
        rnd = self.np_random.uniform()
        if rnd < self.scramble_prob:
            reward = self.reward_per_timestep
            self.state['rot'] = self.np_random.choice([0,1,2,3])

            # mov = MOVE[self.state['rot']]
            # loc = self.state['loc']
            # new_loc = loc + mov

            # if (self.check_inside_area(new_loc) and self.check_walkable(new_loc)):
            #     pass
            # else:
            #     new_loc = loc
            # self.state['loc'] = new_loc

        elif action < 4: 
            if action == 0:
                reward = self.reward_per_timestep
            elif action == 1:
                # Move to adjacent field
                mov = MOVE[self.state['rot']]
                loc = self.state['loc']
                new_loc = loc + mov

                if (self.check_inside_area(new_loc) and self.check_walkable(new_loc)):
                    reward = self.reward_per_timestep
                else:
                    reward = self.reward_wrong_movement
                    new_loc = loc
                self.state['loc'] = new_loc

            elif action == 2:
                reward = self.reward_per_timestep
                self.state['rot'] = (self.state['rot'] + 1) % 4
            elif action == 3:
                reward = self.reward_per_timestep
                self.state['rot'] = (self.state['rot'] - 1) % 4
        # Pickup / Dropoff
        elif action == 4:
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
            
    def create_observation(self, action, loc=None, rot=None, layout=None):

        if loc is None:
            loc = self.state['loc']
        if rot is None:
            rot = self.state['rot']
        if layout is None:
            layout = self.walkable

        if self.image_obs:
            raise NotImplementedError("Image not yet implemented")
        else: 
            obs = np.zeros(288)
            # Remove wall from state
            x = loc[0] - 1 
            y = loc[1] - 1 
            idx = rot * 72 + self.state['pas'] * 36 + y * 6 + x
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