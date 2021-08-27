import json


import numpy as np
import torch

from autorl4do.explanations.util import append_to_json_file


class PolicyWrapper():
    # wrapper for simpler interface to sb3 and d3rl trained_models

    def __init__(self):
        self.alg = None

    def set_env(self, env):
        self.env = env

    def set_model(self, model):
        self.model = model
        self.model_class = self.model.__class__.__name__

    def predict(self, x, state=None, deterministic=True):
        # x = self.to_numpy(x)

        if self.lib == 'sb':  # predicting with sb3
            pred = self.model.predict(x, deterministic=True)
        elif self.lib == 'd3':  # predicting with d3rl
            pred = self.model.predict(x)

        if len(pred) == 2:  # making sure to return a tuple of two elements, as is done in sb3
            pred = pred[0]

        return pred, None

    def predict_value(self, x, action_space):
        tensor_obs = self.to_torch(x)

        if self.lib == 'sb':
            torch_obs = self.to_torch(x)
            if self.alg == 'ppo':
                _, state_val, _ = self.model.policy.forward(torch_obs)
            elif self.alg == 'dqn':
                state_val = max(self.model.policy.q_net(torch_obs).squeeze())
            elif self.alg == 'sac':
                best_action, _ = self.model.policy.predict(tensor_obs)
                state_val = self.model.policy.critic.forward(tensor_obs, torch.tensor(best_action))[0].item()

        elif self.lib == 'd3':
            Q_vals = []
            for act in range(action_space.n):
                tensor_act = self.to_torch([act])
                Q_vals.append(self.model.predict_value(tensor_obs, tensor_act))

            state_val = max(Q_vals)

        return state_val

    def get_Q_vals(self, state, action_space):
        tensor_obs = self.to_torch(state)
        if self.lib == 'sb':
            if self.alg == 'dqn':
                tensor_obs = self.to_torch(state)
                Q_vals = self.model.policy.q_net(tensor_obs).squeeze()
            elif self.alg == 'ppo':
                uncertainty = self.model.policy.forward(tensor_obs)[2].item()
                return uncertainty
            elif self.alg == 'sac':
                Q_vals = []
                best_action, _ = self.model.policy.predict(tensor_obs)
                Q_a = self.model.policy.critic.forward(tensor_obs, torch.tensor(best_action))[0].item()
                Q_vals.append(Q_a)
                for i in range(1000):
                    a = action_space.sample()
                    Q_a = self.model.policy.critic.forward(tensor_obs, torch.tensor([a]))[0].item()
                    Q_vals.append(Q_a)
                Q_vals = torch.tensor(Q_vals)
        else:
            pass  # TODO: do for d3rl

        return Q_vals

    def get_action_certainty(self, state, action_space):
        tensor_obs = self.to_torch(state)
        if self.alg == 'dqn' or self.alg == 'sac':
            Q_vals = self.get_Q_vals(state, action_space)
            certainty = max(torch.softmax(Q_vals, dim=-1))
            return certainty
        elif self.alg == 'ppo':
            uncertainty = self.model.policy.evaluate_actions(tensor_obs, actions=torch.tensor([[]]))[-1].item()
            return 1 - uncertainty

    def load(self, config_path, model_name, env):
        with open(config_path, 'r+') as f:
            data = json.load(f)
            model_params = {}
            for dictionary in data:
                if dictionary['model_name'] == model_name:
                    model_params = dictionary
                    break

            self.model_class = model_params['model_class']
            model_path = model_params['path']
            self.lib = model_params['lib']

            training_params = model_params['training_params']
            training_params['env'] = self.env
            del training_params['lib']  # remove lib key as it's not a training param

            # create instance of class
            alg_class = globals()[self.model_class]
            model = alg_class(env=self.env,
                              batch_size=training_params['batch_size'],
                              gamma=training_params['gamma'],
                              n_critics=training_params['n_critics'])

            if self.lib == 'sb':
                model.load(model_path) # loading sb3 model
            elif self.lib == 'd3':
                model.build_with_env(env)
                model.load_model(model_path + '.pt') # loading d3rl model

            self.model = model

    def save(self, training_params, config_path, path, model_name):
        params = {
            'model_name': model_name,
            'model_class': self.model.__class__.__name__,
            'path': path,
            'lib': training_params['lib']
            }

        self.lib = params['lib']

        # add model parameters
        params['training_params'] = training_params

        # save model parameters
        append_to_json_file(config_path, params)

        # save model
        if self.lib == 'sb':
            # saving sb3 model
            self.model.save(path)
        elif self.lib == 'd3':
            # saving d3rl model
            self.model.save_model(path + '.pt')

    def to_numpy(self, x):
        np_x = np.array(x)
        if np_x.ndim == 1:
            np_x = np.expand_dims(np_x, axis=0)

        return np_x

    def to_torch(self, x):
        tensor_x = torch.Tensor(x)
        if len(tensor_x.shape) == 1:
            tensor_x = tensor_x.unsqueeze(0)

        return tensor_x