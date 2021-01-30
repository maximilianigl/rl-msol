#%%
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import math
import os
import gym
from gym.envs.registration import register
import environments
from cmd import Cmd
import numpy as np
from hierarchical_policy import HierarchicalPolicy
from envs import make_env
# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from subproc_multitask_vec_env import MTSubprocVecEnv
import torch
from curses import wrapper
import curses
from envs import WrapPyTorch

from utils import get_docs, db_uri, db_name
from utils import get_file_id, save_file_from_db
import pprint

MAP_ACTION_TO_KEY = {
    0: 0,
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN"
}


TITLE_MAP = {
    'Top': 'Upper\nRoom',
    'Right' : 'Right\nRoom',
    'Bottom' : 'Lower\nRoom',
    'Left' : 'Left\nRoom'
}


COLOR_MAP = {
    0: (255,196,0), # Yellow
    1: (2,62,255), # Blue
    2: (232,0,11), # Red
    3: (26,201,56), # Green
    4: (241,76,193), # Pink
    5: (0,0,0), # Black
    6: (0,0,0) # Black
}

GREY = np.array([0.5, 0.5, 0.5])

ARROW_MAP = {
    0: (0,0), # Red
    1: (0, -0.0001), # Yellow
    2: (0, 0.0001), # Pink
    3: (0.0001, 0), # Blue
    4: (-0.0001, 0), # Green
    5: (0,0), # Black
    6: (0,0) # Black
}

x_offset_1 = 4
x_offset_2 = 40

y_offset_1 = 0
y_offset_2 = 7
y_offset_3 = 15
y_offset_4 = 25


######################### Plot colormap of options
def getOptionsColormap(room, constraint, envs, directory, config, 
    num_tasks, num_processes_per_task, num_options, type='png', do_master_map=False, do_action_map=False, do_termination_map=False, do_posterior_termination_map=False,
    task_id_reorder=list(range(10))):
    print("Creating Option Colormap...")
    # Set the situation
    task_id = 1

    # valid_coords = np.transpose(np.nonzero(room.walkable))

    # Not dependent on option_id
    option_maps = np.zeros(shape=(num_tasks,) + room.walkable.shape + (3,)) 
    wall_maps = np.ones(shape= room.walkable.shape + (3,))
    posterior_termination_maps = np.zeros(shape=(num_tasks, num_options) + room.walkable.shape + (1,))

    action_maps = np.zeros(shape=(num_options,) + room.walkable.shape + (3,))
    arrow_maps = np.zeros(shape=(num_options,) + room.walkable.shape + (2,))
    arrow_size = np.zeros(shape=(num_options,) + room.walkable.shape + (1,))
    termination_prob_map = np.zeros(shape=(num_options,) + room.walkable.shape + (1,))

    posterior_action_maps = np.zeros(shape=(num_tasks, num_options,) + room.walkable.shape + (3,))
    posterior_arrow_maps = np.zeros(shape=(num_tasks, num_options,) + room.walkable.shape + (2,))
    posterior_arrow_size = np.zeros(shape=(num_tasks, num_options,) + room.walkable.shape + (1,))

    for option_id in range(num_options):
        active_option = torch.full((num_tasks, num_processes_per_task, 1), option_id).long()
        for coords in np.ndindex(room.walkable.shape):
            if not room.walkable[tuple(coords)]:
                wall_maps[tuple(coords)] = GREY
                for task_id in range(num_tasks):
                    option_maps[task_id][tuple(coords)] = GREY
                continue
            # print('_____________')
            # print(coords)
            obs = room.create_observation(layout=room.walkable, loc=coords, action=0)
            # obs = obs.transpose(2, 0, 1) # Channel dimension 2=>0
            obs = np.expand_dims(obs, 0) # Create first dimension for batch
            obs = np.repeat(obs, num_tasks, 0) # Same dimension for each task

            torch_obs = torch.from_numpy(
                obs.reshape(num_tasks, num_processes_per_task, *envs.observation_space.shape)).float()

            z_probs, action_prior_probs, b_prior_probs, b_probs, a_probs = runPolicy(torch_obs, active_option, task_id)
            z_probs = z_probs.numpy()[:,0,:] # Only one process per task
            
            # Color map for master policy
            # Master is independent of last option
            if do_master_map:
                for task_id, z_prob in enumerate(z_probs):
                    # task_id = task_id_reorder[task_id]
                    task_z = np.argmax(z_prob)
                    probability = z_prob[task_z]
                    second_prob = np.partition(np.array(z_prob).flatten(), -2)[-2]
                    option_maps[task_id][tuple(coords)] = np.array(COLOR_MAP[task_z])/255. * (probability - second_prob)
            
            if do_action_map or do_termination_map or do_posterior_termination_map:
                for task_id, b_prob in enumerate(b_probs):
                    # task_id = task_id_reorder[task_id]
                    posterior_termination_maps[task_id][option_id][tuple(coords)] = b_prob[0,1]
                for task_id, a_prob in enumerate(a_probs):
                    a_prob = a_prob[0]
                    prior_action = np.argmax(a_prob).item()
                    probability = a_prob[prior_action].item()
                    second_prob = np.partition(np.array(a_prob).flatten(), -2)[-2]
                    posterior_action_maps[task_id][option_id][tuple(coords)] = np.array(COLOR_MAP[prior_action])/255.
                    posterior_arrow_maps[task_id][option_id][tuple(coords)] = np.array(ARROW_MAP[prior_action])
                    posterior_arrow_size[task_id][option_id][tuple(coords)] = math.sqrt(max(probability - second_prob,0))
                # Color map for option prior
                prior_action = np.argmax(action_prior_probs)
                probability = action_prior_probs[prior_action]
                second_prob = np.partition(np.array(action_prior_probs).flatten(), -2)[-2]
                action_maps[option_id][tuple(coords)] = np.array(COLOR_MAP[prior_action])/255.  # For choosing color of arrow
                arrow_maps[option_id][tuple(coords)] = np.array(ARROW_MAP[prior_action])  # Direction of arrow
                arrow_size[option_id][tuple(coords)] = math.sqrt(max(probability - second_prob,0)) # Size of arrow

                termination_prob_map[option_id][tuple(coords)] = b_prior_probs[1]


    ############################### Plot Maps ####################
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    title = get_print_name(config)

    directory = os.path.join("./image_maps", folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        # Create only starting option Map
    if do_master_map:

        nr_cols = math.ceil(math.sqrt(num_tasks)) 
        nr_rows = math.ceil(num_tasks / nr_cols)

        figsize = (3 * nr_cols, 4 * nr_rows)
        fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, figsize=figsize, squeeze=False)
        fig.suptitle(title)
        for task_id in range(num_tasks):
            col = task_id % nr_cols
            row = task_id // nr_cols
            img = option_maps[task_id]
            # resize_factor = 20 
            # image = resize(img, (img.shape[0] * resize_factor, img.shape[1] * resize_factor), order=0, mode='constant')
            axs[row, col].axis('off')
            axs[row, col].imshow(img.transpose(1,0,2), extent=(0,11,0,11))
            # axs[row, col].set_title(TITLE_MAP[constraint[task_id_reorder[task_id]]], fontsize=30)
            axs[row, col].set_title(constraint[task_id], fontsize=30)
            axs[row, col].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)
            # axs[row, col].set_xlim(0, 11)
            # axs[row, col].set_ylim(0, 11)

        filename = os.path.join(directory, "Option_map_{}.{}".format(_id, type))
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.savefig(filename)


    nr_cols = int(math.sqrt(num_options)) 
    nr_rows = num_options // nr_cols
    figsize = (4.5 * nr_cols, 4.5 * nr_rows)

    resize_factor = 10
    x_size, y_size = room.walkable.shape

    dx = 1.
    dy = 1.

    s = dx/2.
    d = 0.0001

    # Print action map
    if do_action_map:
        fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, sharex=False, sharey=False, figsize=figsize, squeeze=False)
        fig.suptitle(title)

        for opt_id in range(num_options):
            col = opt_id % nr_cols
            row = opt_id // nr_cols

            img = np.copy(wall_maps)
            img[0,0] = COLOR_MAP[opt_id]
            # image = resize(img, (img.shape[0] * resize_factor, img.shape[1] * resize_factor),
            # order=0, mode='constant')
            axs[row, col].imshow(img.transpose(1,0,2))
            axs[row, col].axis('off')
            axs[row, col].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)
            # axs[row, col].set_title("Option {}".format(opt_id))
            for x in range(x_size):
                for y in range(y_size):
                    size_factor = arrow_size[opt_id, x, y, 0]
                    Dx, Dy = arrow_maps[opt_id, x, y]
                    h_width = dx * 0.66 * size_factor
                    h_length = 1.5 * h_width

                    x_corr = 0
                    y_corr = 0

                    if room.walkable[x,y] == 0 or (Dx == 0 and Dy == 0):
                        continue

                    if Dx != 0:
                        x_corr = h_length / 2 * np.sign(Dx)
                    else: 
                        y_corr = h_length / 2 * np.sign(Dy)
                    # print(dx, dy, Dx, Dy, y_corr, x_corr, h_width, h_length)
                    color = tuple(action_maps[opt_id, x, y])
                    axs[row, col].arrow(
                        x = x*dx + dx/2. - x_corr - 0.5,
                        y = y*dy + dy/2. - y_corr - 0.5,
                        dx = Dx,
                        dy = Dy,
                        # fc='k',
                        # ec='k',
                        fc=color,
                        ec=color,
                        head_width=h_width,
                        head_length=h_length,
                        length_includes_head=False)
        
        filename = os.path.join(directory, "Action_map_{}.{}".format(_id, type))
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9)
        fig.savefig(filename)

        # Plot termination probabilities
    if do_termination_map:
        fig, axs = plt.subplots(nrows=nr_rows, ncols=nr_cols, sharex=False, sharey=False, figsize=figsize, squeeze=False)
        fig.suptitle(title)

        for opt_id in range(num_options):
            col = opt_id % nr_cols
            row = opt_id // nr_cols

            img = wall_maps
            # image = resize(img, (img.shape[0] * resize_factor, img.shape[1] * resize_factor),
            # order=0, mode='constant')
            axs[row, col].imshow(img.transpose(1,0,2))
            axs[row, col].axis('off')
            axs[row, col].tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False)
            # axs[row, col].set_title("Option {}".format(opt_id))
            # patches = []
            for x in range(x_size):
                for y in range(y_size):
                    probability = termination_prob_map[opt_id, x, y, 0]
                    size_factor = math.sqrt(probability)

                    x_pos = x*dx + dx/2. - 0.5
                    y_pos = y*dy + dy/2. - 0.5

                    r = 0.5 * size_factor

                    if room.walkable[x,y] == 0:
                        continue

                    circle = Circle((x_pos, y_pos), r, 
                        color=np.array([1.,1.,1.]) * (1 - probability))

                    axs[row, col].add_patch(circle)
                    axs[row, col].annotate("{}".format(str(probability)[:4]), (x_pos, y_pos))
        
        filename = os.path.join(directory, "Termination_map_{}.{}".format(_id, type))
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9)
        fig.savefig(filename)


    if do_posterior_termination_map:
        figsize = (4.5 * num_options, 4.5 * num_tasks)

        fig, axs = plt.subplots(nrows=num_tasks, ncols=num_options, sharex=False, sharey=False, figsize=figsize, squeeze=False)
        fig.suptitle(title)

        for col in range(num_options):
            for row in range(num_tasks):

                img = wall_maps
                # image = resize(img, (img.shape[0] * resize_factor, img.shape[1] * resize_factor),
                # order=0, mode='constant')
                axs[row, col].imshow(img.transpose(1,0,2))
                axs[row, col].axis('off')
                # axs[row, 0].set_ylabel(TITLE_MAP[constraint[task_id_reorder[row]]], fontsize=20)
                # axs[row, 0].set_ylabel(TITLE_MAP[constraint[row]], fontsize=20)
                axs[row, 0].set_ylabel(constraint[row], fontsize=20)
                axs[row, col].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False)
                for x in range(x_size):
                    for y in range(y_size):
                        probability = posterior_termination_maps[row, col, x, y, 0]
                        size_factor = math.sqrt(probability)

                        x_pos = x*dx + dx/2. - 0.5
                        y_pos = y*dy + dy/2. - 0.5

                        r = 0.5 * size_factor

                        if room.walkable[x,y] == 0:
                            continue

                        circle = Circle((x_pos, y_pos), r, 
                            color=np.array([1.,1.,1.]) * (1 - probability))

                        axs[row, col].add_patch(circle)
                        axs[row, col].annotate("{}".format(str(probability)[:4]), (x_pos, y_pos))
            
        filename = os.path.join(directory, "Post_termination_map_{}.{}".format(_id, type))
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9)
        fig.savefig(filename)
    
        ################# Posterior action map
        figsize = (4.5 * num_options, 4.5 * num_tasks)

        fig, axs = plt.subplots(nrows=num_tasks, ncols=num_options, sharex=False, sharey=False, figsize=figsize, squeeze=False)
        fig.suptitle(title)

        ######### Old
        ############ Old

        for col in range(num_options):
            for row in range(num_tasks):

                img = wall_maps
                # image = resize(img, (img.shape[0] * resize_factor, img.shape[1] * resize_factor),
                # order=0, mode='constant')
                axs[row, col].imshow(img.transpose(1,0,2))
                axs[row, col].axis('off')
                # axs[row, 0].set_ylabel(TITLE_MAP[constraint[task_id_reorder[row]]], fontsize=20)
                # axs[row, 0].set_ylabel(TITLE_MAP[constraint[row]], fontsize=20)
                axs[row, 0].set_ylabel(constraint[row], fontsize=20)
                axs[row, col].tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labelleft=False)

                for x in range(x_size):
                    for y in range(y_size):
                        size_factor = posterior_arrow_size[row, col, x, y, 0]
                        Dx, Dy = posterior_arrow_maps[row, col, x, y]
                        h_width = dx * 0.66 * size_factor
                        h_length = 1.5 * h_width

                        x_corr = 0
                        y_corr = 0

                        if room.walkable[x,y] == 0 or (Dx == 0 and Dy == 0):
                            continue

                        if Dx != 0:
                            x_corr = h_length / 2 * np.sign(Dx)
                        else: 
                            y_corr = h_length / 2 * np.sign(Dy)
                        # print(dx, dy, Dx, Dy, y_corr, x_corr, h_width, h_length)
                        color = tuple(posterior_action_maps[row, col, x, y])
                        axs[row, col].arrow(
                            x = x*dx + dx/2. - x_corr - 0.5,
                            y = y*dy + dy/2. - y_corr - 0.5,
                            dx = Dx,
                            dy = Dy,
                            # fc='k',
                            # ec='k',
                            fc=color,
                            ec=color,
                            head_width=h_width,
                            head_length=h_length,
                            length_includes_head=False)
                # for x in range(x_size):
                #     for y in range(y_size):
                #         probability = posterior_termination_maps[row, col, x, y, 0]
                #         size_factor = math.sqrt(probability)

                #         x_pos = x*dx + dx/2. - 0.5
                #         y_pos = y*dy + dy/2. - 0.5

                #         r = 0.5 * size_factor

                #         if room.walkable[x,y] == 0:
                #             continue

                #         circle = Circle((x_pos, y_pos), r, 
                #             color=np.array([1.,1.,1.]) * (1 - probability))

                #         axs[row, col].add_patch(circle)
                #         axs[row, col].annotate("{}".format(str(probability)[:4]), (x_pos, y_pos))
            
        filename = os.path.join(directory, "Post_action_map_{}.{}".format(_id, type))
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9)
        fig.savefig(filename)
    plt.close()


def runPolicy(torch_obs, active_option, task_id):
    active_b = torch.ones_like(active_option)
    active_mask = torch.ones_like(active_option).float()

    with torch.no_grad():
        b, b_log_prob, b_probs = hierarchical_actor_critic.executePolicy(
                obs=torch_obs,
                z=active_option,
                policy_type="termination",
                masks=active_mask,
        )
        z, z_log_prob, z_probs = hierarchical_actor_critic.executePolicy(
                obs=torch_obs,
                z=active_option,
                policy_type="master",
                b=active_b
        )
        action, action_log_prob, a_probs = hierarchical_actor_critic.executePolicy(
                obs=torch_obs,
                z=active_option,
                policy_type="option"
        )

        action_prior_probs = [] 
        # print(torch_obs[:,:,0])
        # print(active_option)
        for i in range(envs.action_space.n):
            action_prior_log_prob = hierarchical_actor_critic.evaluatePrior(
                obs=torch_obs,
                z=active_option, 
                action=torch.full((num_tasks, num_processes_per_task, 1), i),
                policy_type="option"
            )
            action_prior_probs.append(torch.exp(action_prior_log_prob)[task_id, 0].item())
        # print(action_prior_probs)
        b_prior_probs = []
        for i in range(2):
            b_prior_log_prob = hierarchical_actor_critic.evaluatePrior(
                obs=torch_obs,
                z=active_option, 
                action=torch.full((num_tasks, num_processes_per_task, 1), i),
                policy_type="distilled-termination",
                masks=active_mask
            )
            b_prior_probs.append(torch.exp(b_prior_log_prob)[task_id, 0].item())
    return z_probs, action_prior_probs, b_prior_probs, b_probs, a_probs

def load_config(_id):

    docs = get_docs(db_uri, db_name, 'runs')
    doc = docs.find_one({'_id': _id})
    config = doc['config']
    config.update({'num_processes': len(config['tasks']), 'cuda': False})
    return doc, config

def create_environment(environment, add_timestep, tasks, seeds, seed):
    num_tasks = len(tasks)
    print("Creating environments...")
    print("Environment name: {}".format(environment))
    # env = make_env('TwoRooms-v2', seed=0, rank=0, add_timestep=False)
    envs = [make_env(environment, seed, i, add_timestep)
            for i in range(num_processes_per_task * num_tasks)] 
    envs = MTSubprocVecEnv(envs)

    # TODO: Replace with info dict!
    constraint = []
    start_constraint = []
    for task in tasks:
        constraint += [task]*num_processes_per_task
        start_constraint += [False] * num_processes_per_task

    seeds = envs.draw_and_set_task(constraint=constraint, seed=seeds)
    return envs, constraint

def create_and_load_model(_id, num_tasks, alpha, envs, loss, architecture, name, doc, option_init):

    print("Creating model...")
    hierarchical_actor_critic = HierarchicalPolicy(
        num_tasks=num_tasks, 
        num_processes_per_task=num_processes_per_task, 
        alpha=alpha,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space, 
        loss=loss,
        architecture=architecture,
        option_init=option_init)

    print("Loading model parameters from db: {}/{}".format(_id, name))
    file_id = get_file_id(doc=doc, file_name=name)
    save_file_from_db(file_id=file_id, destination='model_tmp.pyt', db_uri=db_uri, db_name=db_name)

    # fname = "/Users/greg/Documents/rl/treeqn/results/57/model_iteration_343750"

    # state_dict  = torch.load(fname, map_location=lambda storage, loc: storage)
    state_dict = torch.load("model_tmp.pyt", map_location=lambda storage, loc: storage)
    hierarchical_actor_critic.load_state_dict(state_dict)
    os.remove('model_tmp.pyt')
    print("Loading model parameters complete.")
    return hierarchical_actor_critic

def get_print_name(config):
    """
    Append the alg_name name with conditions UNLESS they are in the axis_condition
    """
    conditions = config['meta']['conditions'].split(",")
    print_name = ""
    from functools import reduce
    for condition in conditions:
        condition_name = condition.split(".")[-1].replace("_", "-")
        # insert_dict[condition_name] = \
        condition_value = str(reduce(dict.get,
                                     condition.split("."),
                                     config)).replace("_", "-")
        print_name += "-" + condition_name + ":" + condition_value
    return print_name

from environments.grid_room_layouts import ONE_ROOM, TWO_ROOMS, SIX_ROOMS, FOUR_ROOMS, FOUR_LARGE_ROOMS
from environments.grid_room_layouts import NINE_ROOMS, FOUR_ROOMS_XL
from environments.grid_rooms import GridRooms
from environments.taxi import TAXI_ROOMS_LAYOUT, Taxi
# Yellow, Blue, Red, Green
ids = [8249]
exp_name = "0610-msol-MyTaxi-v-1"
m_name = "model_after_training"
# m_name = "final_model"
# seeds = doc['info']['last_training_task_seed']

rooms_kwargs = {}
room = Taxi(image_obs=False)
for s in [False, True]:
    room.state['pas'] = s 
    for _id in ids:
        # _id = 1229
        s_name = "dropoff" if s else "pickup"
        print("Processing _ids: {}".format(_id))
        folder = "-".join([exp_name, s_name, m_name])

        doc, config = load_config(_id)

        seeds = [None] * config['num_processes']
        config_string = pprint.pformat(config)
        num_options = config['architecture']['num_options']
        num_tasks = len(config['tasks'])
        num_processes_per_task = 1


        envs, constraint = create_environment(
            environment=config['environment'],
            add_timestep=config['add_timestep'],
            tasks=config['tasks'],
            seeds=seeds,
            seed=config['seed'])

        hierarchical_actor_critic = create_and_load_model(
            _id=_id,
            num_tasks=num_tasks,
            alpha=config['alpha'],
            envs=envs,
            loss=config['loss'],
            architecture=config['architecture'],
            name=m_name,
            doc=doc,
            option_init=config['option_init'])

        getOptionsColormap(
            room=room,
            constraint=constraint,
            envs=envs,
            directory=folder,
            num_tasks=num_tasks,
            num_processes_per_task=num_processes_per_task,
            num_options=num_options,
            config=config,
            type="png",
            task_id_reorder=[2, 1, 0, 3] + list(range(4,10)),
            do_master_map=True,
            do_action_map=True,
            do_termination_map=True,
            do_posterior_termination_map=True)
