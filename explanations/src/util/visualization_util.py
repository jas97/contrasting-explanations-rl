import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from explanations.src.policy_comparison import get_pref_trajectories


def plot(data, x, y, hue, title, x_label, y_label):
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, ci='sd')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()


def get_num_disagreements(modelA, modelB, env, max_traj_len):
    ''' Returns an average number of disagreements between 2 models '''
    dis_num = []
    for i in range(1):
        disagreement_states, _, _ = get_pref_trajectories(modelA, modelB, env, max_traj_len)
        dis_num.append(len(disagreement_states))

    return dis_num


def get_num_actions(env, model_A, goal_action=1):
    ''' Returns average of times goal action is executed by policy '''
    actions = []
    for i in range(1000):
        done = False
        obs = env.reset()
        num_actions = 0
        while not done:
            action, _ = model_A.predict(obs)
            if action == goal_action:
                num_actions += action

            obs, rew, done, _ = env.step(action)

        actions.append(num_actions)

    print('Average actions: {}'.format(np.mean(actions)))
    return np.mean(actions)

def get_feature_average(env, model, n_ep=100, f_id=0, feature_name=''):
    ''' Returns average value of features '''
    values = []
    for i in range(n_ep):
        done = False
        obs = env.reset()
        ep_vels = []
        while not done:
            action, _ = model.predict(obs)
            feature_val = obs[f_id]  # get agent's velocity
            obs, rew, done, _ = env.step(action)

            ep_vels.append(feature_val)
        values.append(np.mean(ep_vels))

    print('Average value for feature {}: {}'.format(feature_name, np.mean(np.mean(values))))
    return values


def get_avg_velocity(env, model, n_ep=100):
    ''' Returns average velocity throughout the episode '''
    velocities = []
    for i in range(n_ep):
        done = False
        obs = env.reset()
        ep_vels = []
        while not done:
            action, _ = model.predict(obs)
            vel = obs[3]  # get agent's velocity
            obs, rew, done, _ = env.step(action)
            ep_vels.append(vel)
        velocities.append(np.mean(ep_vels))

    print('Average velocity: {}'.format(np.mean(np.mean(velocities))))
    return velocities


def get_average_dist_to_car(env, model, n_ep=100):
    distances = []
    for i in range(n_ep):
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            if env.merged and env.since_merged == 1:
                car_y = obs[1]
                nonaut_y = obs[6]
                dist = car_y - nonaut_y
                distances.append(dist)

            obs, rew, done, _ = env.step(action)

    print('Average y-distance to other car: {}'.format(np.mean(distances)))
    return distances
