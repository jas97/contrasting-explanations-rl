import copy
import json
import os
from json import JSONDecodeError
from itertools import product

import tensorflow as tf
import random
import numpy as np

from autorl4do.utils.generators import InventoryGenerator
import torch


def generate_dataset(prop):
    gen = InventoryGenerator(prop)
    gen.generate()


def seed_everything(seed=1):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    tf.set_random_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)


def append_to_json_file(path, write_data):
    # append to possibly not empty json list in file
    with open(path, 'a+') as f:
        try:
            f.seek(0)
            s = f.read()
            data = json.loads(s)
        except JSONDecodeError:
            data = []

        if isinstance(write_data, dict):
            data.append(write_data)
        elif isinstance(write_data, list):
            data = data + write_data

        f.seek(0)
        f.truncate(0)
        json.dump(data, f)


def generate_outcome_names(discrete_outcomes):
    possible_combs = list(product(discrete_outcomes, discrete_outcomes, repeat=True))
    outcome_names = []
    for x, y in possible_combs:
        outcome_names.append(str(int(x)) + '-' + str(int(y)))

    return outcome_names


class Trajectory:

    def __init__(self):
        self.traj = []
        self.num_steps = 0

    def add(self, state, action):
        state_copy = copy.deepcopy(state)
        self.traj.append((state_copy, action))
        self.num_steps += 1

    def set_end_state_val(self, val):
        self.end_state_val = val

    def set_start_state_importance(self, val):
        self.start_state_importance = val

    def get_path(self):
        return self.traj

    def print(self):
        s = ''
        for state, action in self.traj:
            s += '[{}, {}] '.format(state, action)

        print(s)




