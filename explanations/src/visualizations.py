import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from explanations.envs.cancer_env.cancer_env import EnvCancer
from explanations.src.policy_comparison import get_pref_trajectories
from explanations.src.policy_wrapper import PolicyWrapper


def plot(data, x, y, hue, title, x_label, y_label):
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue, ci='sd')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()

def get_num_disagreements(modelA, modelB, env, max_traj_len):
    dis_num = []
    for i in range(10):
        disagreement_states, _, _ = get_pref_trajectories(modelA, modelB, env, max_traj_len)
        dis_num.append(len(disagreement_states))

    return dis_num


def get_num_actions(env, model_A):
    actions = []
    for i in range(1000):
        done = False
        obs = env.reset()
        num_actions = 0
        while not done:
            action, _ = model_A.predict(obs)
            num_actions += action
            obs, rew, done, _ = env.step(action)

        actions.append(num_actions)

    print('Average actions: {}'.format(np.mean(actions)))
    return actions


def visualize(task, penalties, max_traj_len, feature_names):
    num_disagreements = []
    average_actions = []
    features = []

    model_A = PolicyWrapper()
    model_B = PolicyWrapper()
    model_A.lib = 'sb'
    model_B.lib = 'sb'
    model_A.alg = 'dqn'
    model_B.alg = 'dqn'

    modelA_path = '../models/{}/experiments/model_A_alt'.format(task)
    modelB_path = '../models/{}/experiments/model_dqn_0_{}'

    for p in penalties:
        print('Penalty: {}'.format(p))
        # load models
        if task == 'cancer':
            envp = EnvCancer(penalty=[0.1, p], transition_noise=0.1)
            try:
                model_A.model = DQN.load(modelA_path)
                model_B.model = DQN.load(modelB_path.format(task, p))
                print('Loaded trained models')
            except FileNotFoundError:
                print('Couldn\'t load models. Training...')
                model_B.model = DQN('MlpPolicy', env=envp, verbose=0)
                model_B.model.learn(total_timesteps=600000)
                model_B.model.save(modelA_path)

            model_rand = PolicyWrapper()
            model_rand.lib = 'sb'
            model_rand.alg = 'dqn'
            model_rand.model = DQN('MlpPolicy', env=envp, verbose=0)
            model_rand.model.learn(1000)

        num_actions_B = get_num_actions(envp, model_B)
        average_actions.append((p, num_actions_B))

        num_dis = get_num_disagreements(model_B, model_A, envp, max_traj_len)
        num_disagreements.append((p, num_dis))

    # define dataframes
    df_num_dis = pd.DataFrame.from_dict({key: pd.Series(val) for key, val in num_disagreements})
    df_num_act = pd.DataFrame.from_dict({key: pd.Series(val) for key, val in average_actions})
    df_num_dis = df_num_dis.melt(var_name='Penalty', value_name='Number of disagreements')
    df_num_act = df_num_act.melt(var_name='Penalty', value_name='Action')

    # plot dataframes
    plot(df_num_dis, 'Penalty', 'Number of disagreements', hue=None, title='Number of disagreements (in 1000 episodes)', x_label='Penalty', y_label='Number of disagreements')
    plot(df_num_act, 'Penalty', 'Action', hue=None, title='Average number of treatments administered', x_label='Penalty', y_label='Average number of treatments administered')
    for f in feature_names:
        pass

def main(task):
    if task == 'cancer':
        feature_names = ['C', 'P', 'Q', 'Qp']
        penalties = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # penalties = [ 0.2]
        max_traj_len = 10
        visualize(task, penalties, max_traj_len, feature_names)

if __name__ == '__main__':
    main(task='cancer')
