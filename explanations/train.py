import argparse
import json

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from explanations.envs.driving.driving_simulator import DrivingSimulator
from explanations.envs.lunar_lander.lunar_lander import LunarLander
from explanations.envs.mountain_car.mountain_car_env import MountainCarEnv
from explanations.experiment import experiment
from explanations.src.util import seed_everything


def train(task, model_temp, min_reward, penalties, exploration_fractions, final_eps, n_timesteps, kwargs):
    print('Running {}'.format(task))
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device = {}'.format(device))

    for p in penalties:
        print('------------------ p = {} ------------------'.format(p))
        model_0_path = model_temp.format(task, p)

        if task == 'driving':
            env_p = DrivingSimulator(reward_weights={'car_distance': 5,  # CHANGED FROM 5
                                                     'goal_distance': 10,
                                                     'dev_from_init_vel': 20, # CHANGED FROM 0!!!
                                                     'turn': 50,  # CHANGED FROM 0!!!
                                                     'acc': 0,
                                                     'progress': p})
        elif task == 'mountain-car':
            env_p = MountainCarEnv(action_penalty=p)
        elif task == 'lunar-lander':
            env_p = LunarLander(main_engine_penalty=p)

        done = False
        for f in exploration_fractions:
            for eps in final_eps:
                for n in n_timesteps:
                    if not done:
                        print('Training f = {}, eps = {}, n = {}'.format(f, eps, n))
                        check_env(env_p)

                        model_0 = DQN('MlpPolicy',
                                      env=env_p,
                                      device=device,
                                      exploration_fraction=f,
                                      exploration_final_eps=eps,
                                      verbose=0,
                                      **kwargs)

                        model_0.learn(total_timesteps=n)
                        print('Finished training')

                        mean_rew, _ = evaluate_policy(model_0, env_p, n_eval_episodes=100, deterministic=True)

                        print('p = {}, f = {}, eps = {}, n = {}, reward = {}'.format(p, f, eps, n, mean_rew))

                        if mean_rew > min_reward:
                            print('Good model for p = {} reward = {}'.format(p, mean_rew))
                            model_0.save(model_0_path)
                            done = True


if __name__ == '__main__':
    tasks = ['driving']
    settings_file_temp = 'settings/{}.json'


    for i, task in enumerate(tasks):
        model_0_path = 'models/{}/experiments/model_dqn_0_{}'
        model_1_path = 'models/{}/experiments/model_dqn_1_{}'

        with open(settings_file_temp.format(task), 'r') as j:
            settings = json.loads(j.read())

        # train(task, model_0_path, settings['min_reward'], settings['penalties'], settings['exploration_fraction'],
        #       settings['final_eps'], settings['n_timesteps'], settings['training_args'][0])
        train(task, model_1_path, settings['min_reward'], settings['penalties'], settings['exploration_fraction'],
              settings['final_eps'], settings['n_timesteps'], settings['training_args'][0])

        # experiment(task, settings['penalties'], settings['max_traj_len'], settings['feature_names'])
