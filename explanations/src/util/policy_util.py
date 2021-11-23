import torch
import numpy as np
from scipy.stats import entropy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_value(policy, x):
    tensor_obs = to_torch(x)
    state_val = max(policy.policy.q_net(tensor_obs).squeeze())

    return state_val


def get_Q_values(policy, state):
    tensor_obs = to_torch(state)
    Q_vals = policy.policy.q_net(tensor_obs).squeeze()

    return Q_vals


def get_action_certainty(policy, state):
    tensor_obs = to_torch(state)

    Q_vals = get_Q_values(policy, state)
    certainty = 1.0*max(torch.softmax(Q_vals, dim=-1))

    # baseline = [1.0/len(Q_vals)] * len(Q_vals)
    # sm = torch.softmax(Q_vals, dim=-1).cpu().detach().numpy()
    # certainty = 1 - entropy(sm, qk=baseline)
    return certainty


def get_simulated_Q_vals(policy, env, n_ep=100):
    Q_vals = []
    for i_ep in range(n_ep):
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs)
            Q_val = max(get_Q_values(policy, obs))
            obs, rew, done, _ = env.step(action)

            Q_vals.append(Q_val)

    return Q_vals


def to_numpy(x):
    np_x = np.array(x)
    if np_x.ndim == 1:
        np_x = np.expand_dims(np_x, axis=0)

    return np_x


def to_torch(x):
    tensor_x = torch.Tensor(x).to(device)
    if len(tensor_x.shape) == 1:
        tensor_x = tensor_x.unsqueeze(0)

    return tensor_x