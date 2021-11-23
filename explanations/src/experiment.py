import argparse
import os

from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

from explanations.envs.driving.driving_simulator import DrivingSimulator
from explanations.src.explain import ExplainGen
from explanations.src.util.util import seed_everything
import warnings

from explanations.src.util.visualization_util import plot


def experiment(task, model_path,  penalties, max_traj_len, feature_names):
    seed_everything()
    warnings.filterwarnings("ignore")

    model_baseline_path = model_path + '/model_dqn_0_0'
    model_0_temp = model_path + '/model_dqn_0_{}'
    model_1_temp = model_path + '/model_dqn_1_{}'

    model_baseline = DQN.load(model_baseline_path)

    exp_gen = ExplainGen(task, feature_names)

    results = []
    rewards = []
    for p in penalties[1:]: # first one is the baseline model
        print('='*80)
        print("Penalty: {}".format(p))
        print('='*80)

        model_0_path = model_0_temp.format(p)
        model_1_path = model_1_temp.format(p)

        if task == 'driving':
            env_p = DrivingSimulator(reward_weights={'car_distance': 5,
                                                     'goal_distance': 10,
                                                     'dev_from_init_vel': 20, # CHANGED FROM 0!!!
                                                     'turn': 50,  # CHANGED FROM 0!!!
                                                     'acc': 0,
                                                     'progress': p})

        model_baseline.set_env(env_p)

        model_0 = DQN.load(model_0_path, env=env_p)
        model_1 = DQN.load(model_1_path, env=env_p)

        # evaluation
        model_A_rew, model_A_stddev = evaluate_policy(model_0, env_p, n_eval_episodes=1000, deterministic=True)
        model_B_rew, model_B_stddev = evaluate_policy(model_1, env_p, n_eval_episodes=1000, deterministic=True)
        rewards.append((str(p), r'$\pi_A$', model_A_rew))
        rewards.append((str(p), r'$\pi_B$', model_B_rew))
        print('Dose penalty: {} alg: model_0 mean reward: {}, stddev: {}'.format(p, model_A_rew, model_A_stddev))
        print('Dose penalty: {} alg: model_1 mean reward: {}, stddev: {}'.format(p, model_B_rew, model_B_stddev))

        print('-------- Comparing baseline model with model_0 with penalty = {} --------'.format(p))
        num_traj_baseline = exp_gen.explain(env_p, model_baseline, model_1,  max_traj_len=max_traj_len, num_episodes=1000, step=None, exp_type='explanations')
        results.append((str(p), 'Model 1 vs Baseline', str(num_traj_baseline)))

        print('-------- Comparing model_0 with penalty = {} with model_1 with penalty = {} --------'.format(p, p))
        num_traj_same = exp_gen.explain(env_p, model_0, model_1, max_traj_len=max_traj_len, num_episodes=1000, step=None, exp_type='explanations')
        results.append((str(p), 'Model 1 vs Model 2', str(num_traj_same)))

        # testing comparisons with much less trained model
        if task == 'cancer' or task == 'driving':
            model_bad = DQN('MlpPolicy', env=env_p, verbose=0)
            model_bad.learn(total_timesteps=10000)

        model_bad_rew, model_bad_stddev = evaluate_policy(model_bad, env_p, n_eval_episodes=100, deterministic=True)
        print('Dose penalty: {} alg: model_bad mean reward: {}, stddev: {}'.format(p, model_bad_rew, model_bad_stddev))

        print('-------- Comparing dose model_0 with penalty = {} with random model --------'.format(p))
        num_traj_bad = exp_gen.explain(env_p, model_1, model_bad, max_traj_len=max_traj_len, num_episodes=1000, step=None, exp_type='explanations')
        results.append((str(p), 'Model 1 vs Bad Model', str(num_traj_bad)))
        rewards.append((str(p), r'$\pi_{rand}$', model_bad_rew))

        del model_0
        del model_1
        del model_bad

    results_df = pd.DataFrame.from_records(results, columns=['Penalty', 'Scenario', 'Number of detected disagreement trajectories'])
    results_df.to_csv('results/{}/num_traj.csv'.format(task))

    rewards_df = pd.DataFrame.from_records(rewards, columns=[r'$\theta_p$', 'Policy', 'Average reward'])
    plot(rewards_df, x=r'$\theta_p$', y='Average reward', hue='Policy', x_label=r'$\theta_p$', y_label='Average reward', title='')

    rewards_df.to_csv('results/{}/avg_rewards.csv'.format(task))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cancer')
    args = parser.parse_args()

    experiment(args.task)
