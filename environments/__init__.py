from gym.envs.registration import register


################ Taxi ########################

# CNN version: Removed for consistency between this and the directional version
# register(
#     id='MyTaxi-v1',
#     entry_point='environments.taxi:Taxi',
#     max_episode_steps=50,
#     kwargs={'add_action_in_obs': False}
# )
# register(
#     id='MyTaxi-v2',
#     entry_point='environments.taxi:Taxi',
#     max_episode_steps=50,
#     kwargs={'add_action_in_obs': True}
# )

# FC version
register(
    id='MyTaxi-v1',
    entry_point='environments.taxi:Taxi',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': False,
            'image_obs': False}
)
register(
    id='MyTaxi-v2',
    entry_point='environments.taxi:Taxi',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': True,
            'image_obs': False}
)

# Start everywhere version
register(
    id='MyTaxi-v3',
    entry_point='environments.taxi:Taxi',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': False,
            'start_everywhere': True,
            'image_obs': False}
)
register(
    id='MyTaxi-v4',
    entry_point='environments.taxi:Taxi',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': True,
            'start_everywhere': True,
            'image_obs': False}
)

################ Taxi-2A ########################

register(
    id='MyTaxi2A-v1',
    entry_point='environments.taxi2a:Taxi2A',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': False,
            'scramble_prob': 0.00,
            'image_obs': False}
)

register(
    id='MyTaxi2A-v2',
    entry_point='environments.taxi2a:Taxi2A',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': True,
            'scramble_prob': 0.00,
            'image_obs': False}
)
register(
    id='MyTaxi2A-v3',
    entry_point='environments.taxi2a:Taxi2A',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': False,
            'scramble_prob': 0.10,
            'image_obs': False}
)
register(
    id='MyTaxi2A-v4',
    entry_point='environments.taxi2a:Taxi2A',
    max_episode_steps=50,
    kwargs={'add_action_in_obs': True,
            'scramble_prob': 0.10,
            'image_obs': False}
)

################ MovementBandits ########################

register(
    id='MovementBandits-v0',
    entry_point='environments.movement_bandits:MovementBandits',
    max_episode_steps=50,
)

register(
    id='MovementBandits-v2',
    entry_point='environments.movement_bandits:MovementBandits',
    max_episode_steps=50,
    kwargs={
        'add_action_in_obs': True
    }
)
