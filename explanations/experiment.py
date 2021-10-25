import argparse
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

from explanations.envs.driving.driving_simulator import DrivingSimulator
from explanations.envs.lunar_lander.lunar_lander import LunarLander
from explanations.envs.mountain_car.mountain_car_env import MountainCarEnv
from explanations.src.explain import ExplainGen
from explanations.src.util import seed_everything
import warnings


def experiment(task, penalties, max_traj_len, feature_names):
    seed_everything()
    warnings.filterwarnings("ignore")
    model_baseline_path = 'models/{}/experiments/model_dqn_0_0'.format(task)
    model_0_temp = 'models/{}/experiments/model_dqn_0_{}'
    model_1_temp = 'models/{}/experiments/model_dqn_1_{}'

    model_baseline = DQN.load(model_baseline_path)

    exp_gen = ExplainGen(task, feature_names)

    results = []
    for p in penalties:
        print('='*80)
        print("Penalty: {}".format(p))
        print('='*80)

        model_0_path = model_0_temp.format(task, p)
        model_1_path = model_1_temp.format(task, p)

        if task == 'driving':
            env_p = DrivingSimulator(reward_weights={'car_distance': 5,
                                                     'goal_distance': 10,
                                                     'dev_from_init_vel': 5, # CHANGED FROM 0!!!
                                                     'turn': 0,  # CHANGED FROM 0!!!
                                                     'acc': 0,
                                                     'progress': p})
        elif task == 'mountain-car':
            env_p = MountainCarEnv(action_penalty=p)
        elif task == 'lunar-lander':
            env_p = LunarLander(main_engine_penalty=p)

        model_baseline.set_env(env_p)

        model_0 = DQN.load(model_0_path, env=env_p)
        model_1 = DQN.load(model_1_path, env=env_p)

        # evaluation
        print('Dose penalty: {} alg: model_0 mean reward: {}, stddev: {}'.format(p, *evaluate_policy(model_0, env_p, n_eval_episodes=1000, deterministic=True)))
        print('Dose penalty: {} alg: model_1 mean reward: {}, stddev: {}'.format(p, *evaluate_policy(model_1, env_p, n_eval_episodes=1000, deterministic=True)))

        print('-------- Comparing baseline model with model_0 with penalty = {} --------'.format(p))
        num_traj_baseline = exp_gen.explain(env_p, model_baseline, model_0, max_traj_len=max_traj_len, num_episodes=100, step=None, exp_type='explanations')
        results.append((str(p), 'Model 1 vs Baseline', str(num_traj_baseline)))

        print('-------- Comparing model_0 with penalty = {} with model_1 with penalty = {} --------'.format(p, p))
        num_traj_same = exp_gen.explain(env_p, model_0, model_1, max_traj_len=max_traj_len, num_episodes=100, step=None, exp_type='explanations')
        results.append((str(p), 'Model 1 vs Model 2', str(num_traj_same)))

        # testing comparisons with much less trained model
        if task == 'cancer' or task == 'driving':
            model_bad = DQN('MlpPolicy', env=env_p, verbose=0)
            model_bad.learn(total_timesteps=1000)

        print('Dose penalty: {} alg: model_bad mean reward: {}, stddev: {}'.format(p, *evaluate_policy(model_bad, env_p, n_eval_episodes=100, deterministic=True)))

        print('-------- Comparing dose model_0 with penalty = {} with random model --------'.format(p))
        num_traj_bad = exp_gen.explain(env_p, model_0, model_bad, max_traj_len=max_traj_len, num_episodes=100, step=None, exp_type='explanations')
        results.append((str(p), 'Model 1 vs Bad Model', str(num_traj_bad)))

        del model_0
        del model_1
        del model_bad

    results_df = pd.DataFrame.from_records(results, columns=['Penalty', 'Scenario', 'Number of preference trajectories'])
    results_df.to_csv('results/{}/num_traj.csv'.format(task))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cancer')
    args = parser.parse_args()

    experiment(args.task)
