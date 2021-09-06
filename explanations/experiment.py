import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from explanations.envs.cancer_env.cancer_env import EnvCancer
from explanations.src.explain import ExplainGen
from explanations.src.policy_wrapper import PolicyWrapper
from explanations.src.util import seed_everything


def experiment(task):
    seed_everything()
    model_A_path = 'models/{}/experiments/model_A_alt'.format(task)
    modelA = PolicyWrapper()
    modelA.lib = 'sb'
    model_id = PolicyWrapper()
    model_id.lib = 'sb'

    if task == 'cancer':
        modelA.alg = 'dqn'
        model_id.alg = 'dqn'
        envA = EnvCancer(penalty=[0.1, 0], transition_noise=0.1, timeout_steps=30)
        feature_names = ['C', 'P', 'Q', 'Qp']
        # penalties = [0.2, 0.5, 1]
        penalties = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        max_traj_len = 10
        num_timesteps0 = 500000
        num_timesteps1 = 600000
        step = None

    try:
        if task == 'cancer':
            modelA.model = DQN.load(model_A_path)
            # model_id.model = DQN.load('models/{}/experiments/model_id'.format(task))
        modelA.set_env(envA)
        model_id.set_env(envA)
        print('Loaded trained models.')
    except FileNotFoundError:
        print('Couldn\'t load trained models. Training models:')
        if task == 'cancer':
            modelA.model = DQN('MlpPolicy', env=envA, verbose=0)
            model_id.model = DQN('MlpPolicy', envA, verbose=0)
        modelA.model.learn(total_timesteps=num_timesteps1)
        modelA.model.save(model_A_path)

        # model_id.model.learn(total_timesteps=num_timesteps0)
        # model_id.model.save('models/{}/experiments/model_id'.format(task))

    baseline = BaselinePolicy(action=1)
    # evaluation
    print('Baselines mean reward: {} stddev: {}'.format(*evaluate_policy(baseline, envA, n_eval_episodes=1000, deterministic=True)))
    print('Model A alg: {} {} timesteps mean reward: {}, stddev: {}'.format(modelA.alg, num_timesteps1, *evaluate_policy(modelA, envA, n_eval_episodes=1000, deterministic=True)))
    # print('Model identical alg: {} {} timesteps mean reward: {}, stddev: {}'.format(model_id.alg, num_timesteps0, *evaluate_policy(model_id, envA, n_eval_episodes=1000, deterministic=True)))

    # testing difference between identical models
    # print('--------- Testing difference between identical models ---------')
    exp_gen = ExplainGen(task, feature_names)
    # exp_gen.explain(envA, modelA, modelA, max_traj_len=max_traj_len, num_episodes=1000, step=step, exp_type='all')
    #
    # # testing difference between different models with same reward function
    # print('--------- Testing difference between different models with same reward function ---------')
    # exp_gen.explain(envA, modelA, model_id, max_traj_len=max_traj_len, num_episodes=1000, step=step, exp_type='explanations')

    for p in penalties:
        print('='*80)
        print("Penalty for dose: {}".format(p))
        print('='*80)
        model_0 = PolicyWrapper()
        model_0.lib = 'sb'
        model_0_path = 'models/{}/experiments/model_dqn_0_{}'.format(task, p)

        model_1 = PolicyWrapper()
        model_1.lib = 'sb'
        model_1_path = 'models/{}/experiments/model_dqn_1_{}'.format(task, p)

        if task == 'cancer':
            model_0.alg = 'dqn'
            model_1.alg = 'dqn'
            env_p = EnvCancer(penalty=[0.1, p], transition_noise=0.1, timeout_steps=30)
            try:
                model_0.model = DQN.load(model_0_path)
                model_1.model = DQN.load(model_1_path)
            except FileNotFoundError:
                model_0.model = DQN('MlpPolicy', env=env_p, verbose=0)
                model_0.model.learn(total_timesteps=num_timesteps1)
                model_0.model.save(model_0_path)

                model_1.model = DQN('MlpPolicy', env=env_p, verbose=0)
                model_1.model.learn(total_timesteps=num_timesteps0)
                model_1.model.save(model_1_path)

        # evaluation
        print('Baselines mean reward: {} stddev: {}'.format(*evaluate_policy(baseline, env_p, n_eval_episodes=1000, deterministic=True)))
        print('Dose penalty: {} alg: model_0 mean reward: {}, stddev: {}'.format(p, *evaluate_policy(model_0, env_p, n_eval_episodes=1000, deterministic=True)))
        print('Dose penalty: {} alg: model_1 mean reward: {}, stddev: {}'.format(p, *evaluate_policy(model_1, env_p, n_eval_episodes=1000, deterministic=True)))

        print('-------- Comparing model_0 with dose penalty = {} with model_0 with dose penalty = 0 --------'.format(p))
        exp_gen.explain(envA, model_0, modelA, max_traj_len=max_traj_len, num_episodes=10000, step=step, exp_type='explanations')

        print('-------- Comparing model_0 with dose penalty = {} with model_1 with dose penalty = {} --------'.format(p, p))
        exp_gen.explain(envA, model_0, model_1, max_traj_len=max_traj_len, num_episodes=10000, step=step, exp_type='explanations')

        # testing comparisons with much less trained model
        model_bad = PolicyWrapper()
        model_bad.lib = 'sb'
        if task == 'cancer':
            model_bad.alg = 'dqn'

            model_bad.model = DQN('MlpPolicy', env=env_p, verbose=0)
            model_bad.model.learn(total_timesteps=50000)

        print('Dose penalty: {} alg: model_bad mean reward: {}, stddev: {}'.format(p, *evaluate_policy(model_bad, env_p, n_eval_episodes=1000, deterministic=True)))

        print('-------- Comparing model_bad with dose penalty = {} with model_0 with dose penalty = {} --------'.format(p, p))
        exp_gen.explain(envA, model_bad, model_0, max_traj_len=max_traj_len, num_episodes=10000, step=step,
                        exp_type='explanations')


class BaselinePolicy():
    def __init__(self, action=1):
        self.action=action
        pass

    def predict(self, obs, state=None, deterministic=True):
        return np.array([self.action]), None


if __name__ == '__main__':
    experiment('cancer')
