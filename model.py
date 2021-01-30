import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_, getOutputDimension



def toOnehot(option, num_options):
    option_onehot = torch.zeros(
        size=list(option.size()[:-1]) +  [num_options],
        device=option.device)
    option_onehot.scatter_(-1, option, 1)
    return option_onehot
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, action_space, architecture):
        super(Policy, self).__init__()

        self.encoder_output_size = architecture['encoder_output_size']
        if action_space.__class__.__name__ == "Discrete":
            self.num_outputs = action_space.n
            self.dist = Categorical(self.encoder_output_size, self.num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.encoder_output_size, num_outputs)
        else:
            raise NotImplementedError
        self.train()

        # self.state_size = self.base.state_size

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, features, deterministic=False):
        dist = self.dist(features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()

        # dist.probs only exists if action space is discrete
        try:
            probs = dist.probs
        except AttributeError:
            probs = None

        return action, action_log_probs, probs

    def evaluate_actions(self, features, action):
        dist = self.dist(features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy

class ValueFunction(nn.Module):
    def __init__(self, architecture):
        super(ValueFunction, self).__init__()
        self.encoder_output_size = architecture['encoder_output_size']
        init_fc_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_fc_(nn.Linear(self.encoder_output_size, 1))
        self.train()

    def get_value(self, features):
        value = self.critic_linear(features)
        return value

class Encoder(nn.Module):
    def __init__(self, obs_shape, architecture):
        super().__init__()

        if architecture['encoder'] in ['cnn', 'cnn3']:
            k_size = architecture['k_size']
            padding = architecture['padding']

            last_width = getOutputDimension(
                getOutputDimension(obs_shape[1], k_size=k_size, padding=padding, stride=1),
                k_size=k_size, padding=padding, stride=1)

            last_height = getOutputDimension(
                getOutputDimension(obs_shape[2], k_size=k_size, padding=padding, stride=1),
                k_size=k_size, padding=padding, stride=1)
            
            if architecture['encoder'] == 'cnn3':
                last_width = getOutputDimension(last_width, k_size=k_size, padding=padding, stride=1)
                last_height = getOutputDimension(last_height, k_size=k_size, padding=padding, stride=1)

            n_elements_last_layer = last_width * last_height * 32
            print("Number elements after CNN: {}".format(n_elements_last_layer))

        from operator import mul 
        from functools import reduce
        self.num_options = architecture['num_options']
        self.output_size = architecture['encoder_output_size']
        init_encoder_ = lambda m: init(m,
                        nn.init.orthogonal_,
                        lambda x: nn.init.constant_(x, 0),
                        nn.init.calculate_gain('relu'))
        
        # Ugly hack to make it work on my computer because orthogonal crashes
        import platform
        if platform.system() == 'Darwin':
            init_encoder_ = lambda m: init(m,
                            nn.init.xavier_normal_,
                            lambda x: nn.init.constant_(x, 0),
                            nn.init.calculate_gain('relu'))

        init_fc_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        
        # Compute number of elements in last layer

        if architecture['encoder'] == 'large-fc':
            self.architecture_type = 'fc'
            num_elements = reduce(mul, obs_shape, 1)
            self.actor_encoder = nn.Sequential(
                Flatten(),
                init_encoder_(nn.Linear(num_elements, 1024)),
                nn.ReLU(),
                init_encoder_(nn.Linear(1024, 256)),
                nn.ReLU(),
                # init_encoder_(nn.Linear(256, self.output_size)),
                # nn.ReLU()
                )
            n_elements_last_layer = 256
        elif architecture['encoder'] == 'fc':
            self.architecture_type = 'fc'
            num_elements = reduce(mul, obs_shape, 1)
            self.actor_encoder = nn.Sequential(
                Flatten(),
                init_encoder_(nn.Linear(num_elements, 512)),
                nn.ReLU(),
                init_encoder_(nn.Linear(512, 256)),
                nn.ReLU(),
                # init_encoder_(nn.Linear(256, self.output_size)),
                # nn.ReLU()
                )
            n_elements_last_layer = 256
        elif architecture['encoder'] == 'tiny-fc':
            self.architecture_type = 'fc'
            num_elements = reduce(mul, obs_shape, 1)
            self.actor_encoder = nn.Sequential(
                Flatten(),
                init_encoder_(nn.Linear(num_elements, 32)),
                nn.ReLU(),
                # init_encoder_(nn.Linear(256, self.output_size)),
                # nn.ReLU()
                )
            n_elements_last_layer = 32
        elif architecture['encoder'] == 'small-fc':
            self.architecture_type = 'fc'
            num_elements = reduce(mul, obs_shape, 1)
            self.actor_encoder = nn.Sequential(
                Flatten(),
                init_encoder_(nn.Linear(num_elements, 64)),
                nn.ReLU(),
                init_encoder_(nn.Linear(64, 64)),
                nn.ReLU(),
                # init_encoder_(nn.Linear(256, self.output_size)),
                # nn.ReLU()
                )
            n_elements_last_layer = 64
        elif architecture['encoder'] == 'medium-fc':
            self.architecture_type = 'fc'
            num_elements = reduce(mul, obs_shape, 1)
            self.actor_encoder = nn.Sequential(
                Flatten(),
                init_encoder_(nn.Linear(num_elements, 256)),
                nn.ReLU(),
                init_encoder_(nn.Linear(256, 128)),
                nn.ReLU(),
                # init_encoder_(nn.Linear(256, self.output_size)),
                # nn.ReLU()
                )
            n_elements_last_layer = 128
        elif architecture['encoder'] == 'cnn':
            self.architecture_type = 'cnn'
            # Computer number of elements in last layer
            num_inputs = obs_shape[0]
            k_size = architecture['k_size']
            padding = architecture['padding']
            self.actor_encoder = nn.Sequential(
                init_encoder_(nn.Conv2d(num_inputs, 32, k_size, stride=1, padding=padding)),
                nn.ReLU(),
                init_encoder_(nn.Conv2d(32, 32, k_size, stride=1, padding=padding)),
                nn.ReLU(),
                Flatten(),
            )
        elif architecture['encoder'] == 'cnn3':
            self.architecture_type = 'cnn'
            num_inputs = obs_shape[0]
            k_size = architecture['k_size']
            padding = architecture['padding']
            self.actor_encoder = nn.Sequential(
                init_encoder_(nn.Conv2d(num_inputs, 32, k_size, stride=1, padding=padding)),
                nn.ReLU(),
                init_encoder_(nn.Conv2d(32, 32, k_size, stride=1, padding=padding)),
                nn.ReLU(),
                init_encoder_(nn.Conv2d(32, 32, k_size, stride=1, padding=padding)),
                nn.ReLU(),
                Flatten(),
            )
        else:
            raise NotImplementedError("Encoder '{}' not implemented".format(architecture['encoder']))
            
        
        # Add the output of option input fc

        self.option_layer = init_fc_(nn.Linear(self.num_options, 128))
        self.actor = nn.Sequential(
            init_encoder_(nn.Linear(n_elements_last_layer + 128, self.output_size)),
            nn.ReLU()
        )

        self.train()
    

    def forward(self, inputs, option):
        norm_factor = 1.0
        if self.architecture_type == 'cnn':
            norm_factor = 255.0

        actor_features = self.actor_encoder (inputs / norm_factor)
        option_features = self.option_layer(toOnehot(option, self.num_options))
        features = torch.cat([actor_features, option_features], dim=1)
        features = self.actor(features)

        return features


