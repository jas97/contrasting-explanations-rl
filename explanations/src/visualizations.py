import os

import pandas as pd
from stable_baselines3 import DQN

from explanations.envs.driving.driving_simulator import DrivingSimulator
from explanations.src.util.visualization_util import get_num_actions, get_avg_velocity, get_average_dist_to_car, \
    get_num_disagreements, plot, get_feature_average


def visualize(task, model_path, penalties, max_traj_len):
    baseline_path = model_path + '/model_dqn_0_0'
    model_path = model_path + '/model_dqn_1_{}'

    num_disagreements = []
    velocities = []
    distances = []

    for p in penalties[0:]: # skip the first one it's baseline
        print('Penalty: {}'.format(p))
        # init environments
        if task == 'driving':
            envp = DrivingSimulator(reward_weights={'car_distance': 5,  # CHANGED FROM 5
                                                     'goal_distance': 10,
                                                     'dev_from_init_vel': 20,  # CHANGED FROM 0!!!
                                                     'turn': 50,  # CHANGED FROM 0!!!
                                                     'acc': 0,
                                                     'progress': p})

        # load models
        model_baseline = DQN.load(baseline_path)
        model = DQN.load(model_path.format(p))

        model_rand = DQN('MlpPolicy', env=envp, verbose=0)
        model_rand.learn(1000)

        if task == 'cancer':
            avg_actions = get_num_actions(envp, model, goal_action=1)
        elif task == 'driving':
            avg_velocity = get_avg_velocity(envp, model)
            avg_dist = get_average_dist_to_car(envp, model)
            velocities.append((p, avg_velocity))
            distances.append((p, avg_dist))

        # num_dis = get_num_disagreements(model_baseline, model, envp, max_traj_len)
        # num_disagreements.append((p, num_dis))

    measures = [{
        'data': velocities,
        'title': '',
        'x-label': 'p',
        'y-label': "Average velocity"},
        {
        'data': distances,
        'title': '',
        'x-label': 'p',
        'y-label': 'Average y-distance to non-autonomous vehicle'
        }]

    # plot number of disagreements over models
    # df_num_dis = pd.DataFrame.from_dict({key: pd.Series(val) for key, val in num_disagreements})
    # df_num_dis = df_num_dis.melt(var_name='Penalty', value_name='Number of disagreements')
    # plot(df_num_dis, 'Penalty', 'Number of disagreements', hue=None, title='Number of disagreements (in 1000 episodes)', x_label='Penalty', y_label='Number of disagreements')

    # plot feature values
    for measure in measures:
        df_num_act = pd.DataFrame.from_dict({key: pd.Series(val) for key, val in measure['data']})
        df_num_act = df_num_act.melt(var_name='var', value_name='value')

        # plot dataframes
        plot(df_num_act, 'var', 'value', hue=None, title=measure['title'],
             x_label=measure['x-label'], y_label=measure['y-label'])



