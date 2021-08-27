from gym.envs.registration import register

register(
    id='EnvCancer-v0',
    entry_point='autorl4do.explanations.cancer_env.cancer_env:EnvCancer',
)