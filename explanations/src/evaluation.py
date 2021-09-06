import json

import numpy as np
from dowhy import CausalModel

from autorl4do.explanations.envs.cancer_env.cancer_env import EnvCancer
import pandas as pd

from autorl4do.explanations.util import seed_everything


def evaluate(env, graph, feature_names, treatment, action_names, expected_rel):
    # gather data for evaluation
    data = []

    for i in range(1000):
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            new_obs, rew, done, _ = env.step(action)
            action_record = np.zeros((len(action_names)))
            if action == 1:
                action_record[0] = 1
            elif action == 2:
                action_record[1] = 1
            elif action == 3:
                action_record[2] = 1

            record = list(obs)
            record += list(action_record)
            data += [record]
            obs = new_obs

    df = pd.DataFrame.from_records(data, columns=feature_names + action_names)
    df[action_names] = df[action_names].astype(bool)
    df = df.dropna()

    # for each feature estimate the effect of treatment
    causal_effects = {}
    for f in feature_names:
        print('Estimating effect of {} on {}'.format(treatment, f))
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=f,
            graph=graph)

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print(identified_estimand)

        causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
        print("Causal Estimate for effect of {} on {} = {}".format(treatment, f, str(causal_estimate.value)))
        causal_effects[f] = np.sign(causal_estimate.value)

    eval = {f: np.equal(eff, expected_rel[f]) for (f, eff) in causal_effects.items()}

    return eval


def main():
    seed_everything()
    env = EnvCancer(penalty=[1, 1])  # reward doesn't matter here

    data = []
    obs = env.reset()
    record = list(obs)
    record.append(0)
    data += [record]
    for i in range(1000):
        done = False
        while not done:
            action = env.action_space.sample()
            new_obs, rew, done, _ = env.step(action)
            record = list(obs) + list(new_obs)
            record.append(action)
            data += [record]
            obs = new_obs

    df = pd.DataFrame.from_records(data, columns=['C_old', 'P_old', 'Q_old', 'Qp_old', 'C_new', 'P_new', 'Q_new', 'Qp_new', 'A'])
    df['A'] = df['A'].astype(bool)
    df = df.dropna()

    print(df.head(10))
    print(df['C_old'].corr(df['Qp_new']))

    model = CausalModel(
        data=df,
        treatment='C_old',
        outcome='Qp_new',
        graph="""graph[directed 1 node[id "A" label "A"]
                        node[id "C_old" label "C_old"]
                        node[id "P_old" label "P_old"]
                        node[id "Q_old" label "Q_old"]
                        node[id "Qp_old" label "Qp_old"]
                        node[id "C_new" label "C_new"]
                        node[id "P_new" label "P_new"]
                        node[id "Q_new" label "Q_new"]
                        node[id "Qp_new" label "Qp_new"]
                        edge[source "C_old" target "C_new"]
                        edge[source "C_new" target "P_new"]
                        edge[source "C_new" target "Q_new"]
                        edge[source "C_new" target "Qp_new"]
                        edge[source "P_old" target "P_new"]
                        edge[source "P_old" target "Q_new"]
                        edge[source "Q_old" target "Q_new"]
                        edge[source "Q_old" target "Qp_new"]
                        edge[source "Qp_old" target "Qp_new"]
                        edge[source "Qp_old" target "P_new"]]"""

    )
    model.view_model()

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression", target_units=lambda x: x['A'] == 1)
    print(causal_estimate)
    print("Causal Estimate is " + str(causal_estimate.value))



if __name__ == '__main__':
    main()