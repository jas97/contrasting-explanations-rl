import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from autorl4do.explanations.envs.cancer_env.cancer_env import EnvCancer
from autorl4do.explanations.policy_comparison import get_pref_trajectories
from autorl4do.explanations.policy_wrapper import PolicyWrapper


def plot(data, x, y, hue, title, x_label, y_label):
    ax = sns.lineplot(data=data, x=x, y=y, hue=hue)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()


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

    return np.mean(actions)


def visualize(task, env, penalties, max_traj_len, feature_names):
    num_disagreements = []
    average_actions = []
    features = []

    model_A = PolicyWrapper()
    model_B = PolicyWrapper()
    model_A.lib = 'sb'
    model_B.lib = 'sb'
    model_A.alg = 'dqn'
    model_B.alg = 'dqn'

    modelA_path = 'models/{}/experiments/model_A_alt'.format(task)
    modelB_path = 'models/{}/experiments/model_dqn_0_{}'

    for p in penalties:
        print('Penalty: {}'.format(p))
        # load models
        if task == 'cancer':
            envp = EnvCancer(penalty=[1, p], transition_noise=0.0)
            try:
                model_A.model = DQN.load(modelA_path)
                model_B.model = DQN.load(modelB_path.format(task, p))
            except FileNotFoundError:

                model_B.model = DQN('MlpPolicy', env=envp, verbose=0)
                model_B.model.learn(total_timesteps=600000)
                model_B.model.save(modelA_path)

            model_rand = PolicyWrapper()
            model_rand.lib = 'sb'
            model_rand.alg = 'dqn'
            model_rand.model = DQN('MlpPolicy', env=envp, verbose=0)
            model_rand.model.learn(1000)

        # num_actions_A = get_num_actions(env, model_A)
        num_actions_B = get_num_actions(env, model_B)

        average_actions.append((p, num_actions_B))

        # disagreement_states, disagreement_traj, disagreement_outcomes = get_pref_trajectories(model_A, model_B, env, max_traj_len)
        disagreement_states, disagreement_traj, disagreement_outcomes = get_pref_trajectories(model_B, model_rand, env, max_traj_len)

        num_disagreements.append((p, len(disagreement_states)))

        # f_means = []
        # for i, f in enumerate(feature_names):
        #     outcome_f_A = [outcome[0][i] for outcome in disagreement_outcomes]
        #     outcome_f_B = [outcome[1][i] for outcome in disagreement_outcomes]
        #     if len(disagreement_outcomes) > 0:
        #         f_means.append(np.mean(outcome_f_A))
        #         f_means.append(np.mean(outcome_f_B))
        #     else:
        #
        #
        # features.append((p, *f_means))

    # define dataframes
    df_num_dis = pd.DataFrame.from_records(num_disagreements, columns=['Penalty',  'Num disagreements'])
    df_num_act = pd.DataFrame.from_records(average_actions, columns=['Penalty',  'Mean actions'])
    # df_features = pd.DataFrame.from_records(features, columns=['Penalty'] + [f + '_A' for f in feature_names] + [f + '_B' for f in feature_names])

    # plot dataframes
    plot(df_num_dis, 'Penalty', 'Num disagreements', hue=None, title='Number of disagreements (in 1000 episodes)', x_label='Penalty', y_label='Number of disagreements')
    # df_num_act = df_num_act.melt(value_vars=['A', 'B'], id_vars=['Penalty'], value_name='Mean actions', var_name='Model')
    plot(df_num_act, 'Penalty', 'Mean actions', hue=None, title='Average number of treatments administered', x_label='Penalty', y_label='Average number of treatments administered')
    for f in feature_names:
        pass

def main(task):
    if task == 'cancer':
        env = EnvCancer(penalty=[1, 0], transition_noise=0.0)
        feature_names = ['C', 'P', 'Q', 'Qp']
        penalties= [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # penalties = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
        max_traj_len = 10
        visualize(task, env, penalties, max_traj_len, feature_names)

if __name__ == '__main__':
    main(task='cancer')
