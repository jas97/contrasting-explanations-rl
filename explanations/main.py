import argparse
from stable_baselines3 import SAC
from stable_baselines3 import PPO

from autorl4do.explanations.config.graphs import GRAPHS
from autorl4do.explanations.envs.cancer_env.cancer_env import EnvCancer
from autorl4do.explanations.envs.driving.driving_simulator import DrivingSimulator
from autorl4do.explanations.envs.inventory.inventory_env import EnvInventory
from autorl4do.explanations.envs.lunar_lander.lunar_lander import LunarLander
from autorl4do.explanations.envs.vehicle_routing.vehicle_routing import VehicleRouting
from autorl4do.explanations.evaluation import evaluate
from autorl4do.explanations.explain import ExplainGen
from autorl4do.explanations.policy_wrapper import PolicyWrapper
from autorl4do.explanations.util import seed_everything


def main():
    seed_everything()

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='inventory')

    args = parser.parse_args()
    task = args.task

    print('Task: {}'.format(task))

    model_A_path = 'models/{}/model_A'.format(task)
    model_B_path = 'models/{}/model_B'.format(task)

    modelA = PolicyWrapper()
    modelB = PolicyWrapper()
    modelA.lib = 'sb'
    modelB.lib = 'sb'

    if task == 'inventory':
        feature_names = ['inventory', 'demand']

        envA = EnvInventory(stock_penalty=1, shipment_penalty=10)
        modelA.model = SAC("MlpPolicy", envA, verbose=1)
        envB = EnvInventory(stock_penalty=1, shipment_penalty=0)
        modelB.model = SAC("MlpPolicy", envB, verbose=1)
        max_traj_len = 10
        step = 10
        treatment = 'A'
    elif task == 'cancer':
        feature_names = ['C', 'P', 'Q', 'Qp']

        envA = EnvCancer(penalty=[1, 0], transition_noise=0.1)
        modelA.model = PPO("MlpPolicy", envA, verbose=1)
        envB = EnvCancer(penalty=[1, 0.3], transition_noise=0.1)
        modelB.model = PPO("MlpPolicy", envB, verbose=1)
        max_traj_len = 10
        step = None
        treatment = 'C'
    elif task == 'driving':
        envA = DrivingSimulator(reward_weights={'car_distance': -100,
                                                'goal_distance': -1,
                                                'dev_from_init_vel': 0,
                                                'turn': -50,
                                                'acc': 0,
                                                'progress': -1})
        modelA.model = SAC("MlpPolicy", envA, verbose=1, learning_starts=40000)
        envB = DrivingSimulator(reward_weights={'car_distance': -1,
                                                'goal_distance': -1,
                                                'dev_from_init_vel': 0,
                                                'turn': -50,
                                                'acc': 0,
                                                'progress': -10})
        modelB.model = SAC("MlpPolicy", envB, verbose=1, learning_starts=40000)
        feature_names = ['x', 'y', 'heading', 'speed', 'steering wheel angle']
        max_traj_len = 10
        step = 0.1
        treatment = ''
    elif task == 'vehicle-routing':
        envA = VehicleRouting(grid=(10, 10), order_miss_penalty=50)  # lower penaly --> should take more risks
        modelA.model = PPO('MlpPolicy', envA, verbose=1)
        envB = VehicleRouting(grid=(10, 10), order_miss_penalty=1000)  # higher penalty --> only pick up orders if sure
        modelB.model = PPO('MlpPolicy', envB, verbose=1)
        feature_names = []
        max_traj_len = 100
        step = None
    elif task == 'lunar-lander':
        envA = LunarLander(main_engine_penalty=0.3, side_engine_penalty=0.03, time_taken_penalty=0)
        modelA.model = PPO('MlpPolicy', envA, verbose=1)
        envB = LunarLander(main_engine_penalty=5, side_engine_penalty=0.03, time_taken_penalty=0)
        modelB.model = PPO('MlpPolicy', envB, verbose=1)
        feature_names = ['x', 'y', 'vel_x', 'vel_y', 'angle', 'ang_vel', 'left_leg', 'right_leg']
        max_traj_len = 30
        step = None
        treatment = 'main_engine'
    else:
        raise ValueError('Task {} is not supported. Supported tasks are inventory, cancer and driving')

    modelA.set_env(envA)
    modelB.set_env(envB)
    try:
        if task == 'cancer' or task == 'lunar-lander':
            modelA.model = PPO.load(model_A_path)
            modelB.model = PPO.load(model_B_path)
            print('Loaded trained models')
        elif task == 'driving' or task == 'inventory':
            modelA.model = SAC.load(model_A_path)
            modelB.model = SAC.load(model_B_path)
            print('Loaded trained models')
    except FileNotFoundError:
        print('Failed loading trained models. Starting training')
        modelA.model.learn(total_timesteps=100000)
        modelB.model.learn(total_timesteps=100000)
        modelA.model.save(model_A_path)
        modelB.model.save(model_B_path)

    obs=envA.reset()
    done = False
    actions = [0,0,0,0]
    while not done:
        action, _ = modelA.model.predict(obs)
        envA.render()
        print('Obs: {}'.format(obs))
        print('Action: {}'.format(action))
        obs, rew, done, _ = envA.step(action)
        print('Reward: {}'.format(rew))
        actions[action] += 1
    envA.close()

    print("Actions: {}".format(actions))

    obs = envB.reset()
    done = False
    actions = [0, 0, 0, 0]
    while not done:
        action, _ = modelB.model.predict(obs)
        envB.render()
        print('Obs: {}'.format(obs))
        print('Action: {}'.format(action))
        obs, rew, done, _ = envB.step(action)
        print('Reward: {}'.format(rew))
        actions[action] += 1
    envB.close()

    print("Actions: {}".format(actions))

    exp_gen = ExplainGen(task, feature_names)
    exp_gen.explain(envA, modelA, modelB, max_traj_len, num_episodes=10000, step=step, exp_type='explanations')

    # eval = evaluate(envA, GRAPHS[task], feature_names, treatment, action_names=['left_engine', 'main_engine', 'right_engine'], expected_rel=expected_rel)
    # print('Evaluation: {}'.format(eval))


if __name__ == '__main__':
    main()