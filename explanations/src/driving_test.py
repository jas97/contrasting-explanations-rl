from stable_baselines3 import SAC, TD3
from stable_baselines3.common.env_checker import check_env

from autorl4do.explanations.envs.driving.driving_simulator import DrivingSimulator
from autorl4do.explanations.policy_wrapper import PolicyWrapper
from autorl4do.explanations.util import seed_everything


def main():
    seed_everything()
    env = DrivingSimulator(reward_weights={'car_distance': -1,
                                            'goal_distance': -1,
                                            'dev_from_init_vel': 0,
                                            'turn': -50,
                                            'acc': 0,
                                            'progress': -5})  # with 1 here is also good

    check_env(env)

    model = PolicyWrapper()
    model.set_env(env)
    model.lib = 'sb'
    model.model = SAC("MlpPolicy", env, verbose=1, learning_starts=40000)
    model.model.learn(total_timesteps=100000)

    obs = env.reset()
    done = False
    env.render()
    while not done:
        action, _ = model.predict(obs)
        action = action.squeeze()

        print('Action: {}'.format(action))
        obs, rew, done, _ = env.step(action)
        print('Reward: {}'.format(rew))
        env.render()


if __name__ == '__main__':
    main()