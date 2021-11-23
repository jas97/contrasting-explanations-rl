
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler

from explanations.src.util.policy_util import predict_value, get_action_certainty, get_simulated_Q_vals
from explanations.src.util.util import Trajectory


def get_pref_trajectories(policyA, policyB, env, max_traj_len, num_episodes=1000):
    disagreement_states, disagreement_traj = gather_contrasting_data(env, policyA, policyB, num_episodes, max_traj_len=max_traj_len)
    print('Total disagreements: {}'.format(len(disagreement_states)))
    if len(disagreement_states) > 0:
        disagreement_traj = [(t1, t2) for t1, t2 in disagreement_traj if (len(t1.traj) > 1) and (len(t2.traj) > 1)]
        disagreement_states = [t1.traj[0][0] for t1, t2 in disagreement_traj]
        disagreement_outcomes = [(t1.traj[-1][0], t2.traj[-1][0]) for (t1, t2) in disagreement_traj]

        Q_A, Q_B,  Q_A_s, Q_B_s = get_Q_vals(policyA, policyB, env, disagreement_states, disagreement_outcomes)
        state_importance = [get_state_importance(s, env, policyA, policyB) for s in disagreement_states]

        scores = [get_traj_score(Q_A[i], Q_B[i], Q_A_s[i], Q_B_s[i], state_importance[i])
                  for i in range(len(disagreement_traj))]

        filtered_traj = [t for i, t in enumerate(disagreement_traj) if scores[i]]

        filtered_dis_states = [t1.traj[0][0] for t1, t2 in filtered_traj]
        filtered_outcomes = [(t1.traj[-1][0], t2.traj[-1][0]) for (t1, t2) in filtered_traj]

        print('Number of preference traj pairs: {}'.format(len(filtered_traj)))
        return filtered_dis_states, filtered_traj, filtered_outcomes
    else:
        return disagreement_states, disagreement_traj, []


def get_Q_vals(policyA, policyB, env, disagreement_states, disagreement_outcomes):
    Q_A = [predict_value(policyA, a) for a, b in disagreement_outcomes]
    Q_B = [predict_value(policyB, b) for a, b in disagreement_outcomes]

    Q_A_s = [predict_value(policyA, a) for a in disagreement_states]
    Q_B_s = [predict_value(policyB, b) for b in disagreement_states]

    scaler_A = MinMaxScaler(feature_range=[0, 1])
    scaler_B = MinMaxScaler(feature_range=[0, 1])

    Q_A_simulated = get_simulated_Q_vals(policyA, env)
    Q_B_simulated = get_simulated_Q_vals(policyB, env)

    scaler_A.fit([[min(Q_A + Q_A_s + Q_A_simulated)], [max(Q_A + Q_A_s +  Q_A_simulated)]])
    scaler_B.fit([[min(Q_B + Q_B_s + Q_B_simulated)], [max(Q_B + Q_B_s + Q_B_simulated)]])

    Q_A = [scaler_A.transform([[q_a]]).item() for q_a in Q_A]
    Q_B = [scaler_B.transform([[q_b]]).item() for q_b in Q_B]

    Q_A_s = [scaler_A.transform([[s]]).item() for s in Q_A_s]
    Q_B_s = [scaler_B.transform([[s]]).item() for s in Q_B_s]

    return Q_A, Q_B, Q_A_s, Q_B_s


def get_state_importance(state, env, policyA, policyB):
    importance_A = get_action_certainty(policyA, state)
    importance_B = get_action_certainty(policyB, state)

    return (importance_A + importance_B).item() / 2


def get_traj_score(Q_A, Q_B, Q_A_s, Q_B_s, state_importance):
    traj_score = 1 - abs(Q_A - Q_B)
    state_disagreement = 1 - abs(Q_A_s - Q_B_s)

    return (traj_score > 0.9) and (state_importance > 0.8) and (state_disagreement > 0.9)


def gather_contrasting_data(env, modelA, modelB, num_episodes=100, max_traj_len=10):
    disagreement_states = []
    disagreement_traj = []

    for i_ep in range(num_episodes):
        done = False
        env.reset()
        while not done:
            obs = env.get_obs()
            actionA, _ = modelA.predict(obs, deterministic=True)
            actionB, _ = modelB.predict(obs, deterministic=True)

            if env.scaler is not None:
                actionA = env.scaler.inverse_transform([actionA]).squeeze()
                actionB = env.scaler.inverse_transform([actionB]).squeeze()

            if np.not_equal(actionA, actionB).any():  # branch along two possible paths
                disagreement_states.append(env.get_state())
                trajA = Trajectory()
                trajB = Trajectory()

                checkpoint = env.get_state()   # save env checkpoint
                num_steps = env.get_steps_elapsed()  # store number of elapsed steps

                trajA.add(checkpoint, actionA)
                trajB.add(checkpoint, actionB)

                trajA, done, end_state = unroll_policy(env, modelA, actionA.squeeze(), trajA, k=max_traj_len)

                env.update_state(checkpoint)  # reset env to the checkpoint
                env.steps_elapsed = num_steps  # reset num steps

                trajB, _, _ = unroll_policy(env, modelB, actionB.squeeze(), trajB, k=max_traj_len)

                disagreement_traj.append((trajA, trajB))

                if not done:
                    env.update_state(end_state)
                    env.set_steps_elapsed(env.get_steps_elapsed() + max_traj_len)  # += k

            else:  # continue along
                obs, reward, done, _ = env.step(actionA)  # either action is fine because they're same

    if len(disagreement_traj) > 0:
        trajectoriesA, trajectoriesB = (zip(*disagreement_traj))
        disagreement_traj = list(zip(trajectoriesA, trajectoriesB))
    else:
        print('No disagreement found between policies')

    return disagreement_states, disagreement_traj


def unroll_policy(env, model, action, traj, k=10):
    # unroll from a specific place in the environment and store transitions in a trajectory
    obs, reward, done, _ = env.step(action)
    end_state = env.get_state()
    count = 1

    if done:
        traj.set_end_state_val(reward)
        return traj, done, end_state

    while count < k and not done:
        count += 1

        action, _ = model.predict(obs, deterministic=True)

        if env.scaler is not None:
            traj.add(env.get_state(), env.scaler.inverse_transform([action]))
        else:
            traj.add(env.get_state(), action)

        obs, reward, done, _ = env.step(action)

        if done or count == k:
            end_state = env.get_state()

    return traj, done, end_state


def get_disagreement_data(pref_traj, modelA, modelB, env, step=None):
    # extracts states from pref_traj where trained_models disagree
    data = []

    if pref_traj is None or len(pref_traj) == 0:
        return data

    for traj_A, traj_B in pref_traj:
        for s, a in traj_A.get_path():
            data = append_disagreement_event(s, modelA, modelB, data, env, step)

        for s, a in traj_B.get_path():
            data = append_disagreement_event(s, modelA, modelB, data, env, step)

            num_features = s.shape[0]

    column_names = []
    dtypes = {}
    dtypes['policy'] = 'category'
    for f in range(num_features):  # feature names are strings '0', '1' ...
        column_names.append(str(f))

    feature_names = column_names.copy()

    target_name = 'action'
    policy_name = 'policy'
    column_names.append(policy_name)
    column_names.append(target_name)

    df = pd.DataFrame.from_records(data, columns=column_names)
    df = df.astype(dtypes)

    feature_schema = {
        'features': feature_names,
        'target': target_name,
        'policy': policy_name,
        'policy_key': len(feature_names)  # policy is after all other features
    }

    return df, feature_schema


def append_disagreement_event(s, modelA, modelB, data, env, step):
    actionA, _ = modelA.predict([s], deterministic=True)
    actionB, _ = modelB.predict([s], deterministic=True)

    if env.scaler is not None:
        actionA = env.scaler.inverse_transform(actionA)
        actionB = env.scaler.inverse_transform(actionB)

    actionA = actionA.item()
    actionB = actionB.item()

    # if actionA.squeeze().shape[0] > 1:
    #     actionA = actionA.item()
    #     actionB = actionB.item()
    # else:  # in case of multiple actions
    #     actionA = actionA.squeeze()
    #     actionB = actionB.squeeze()

    if step is not None:
        actionA = int(actionA / step)
        actionB = int(actionB / step)

    data.append((*list(s.squeeze()), 0, actionA))
    data.append((*list(s.squeeze()), 1, actionB))

    return data


def build_tree(df, feature_schema, output_file, feature_names):
    # builds decision tree from dataframe
    policy_name = feature_schema['policy']
    feature_ids = feature_schema['features'] + [policy_name]
    target_name = feature_schema['target']

    features = df[feature_ids].values
    target = df[[target_name]].values.squeeze()

    clf = tree.DecisionTreeClassifier(min_samples_leaf=100)  # TODO: customize to allow decision and regression
    clf = clf.fit(features, target)
    tree.plot_tree(clf)
    plt.show()

    return clf