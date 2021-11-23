import numpy as np
from scipy import stats


def get_contrastive_explanations(outcomes, env, feature_schema, agent_names, policy_names):
    # perform t-test on each feature of states separately
    explanation = extract_feature_diff(outcomes, feature_schema, agent_names[0], policy_names[0], policy_names[1])

    env.close()
    return explanation


def extract_feature_diff(outcome_pairs, feature_schema, agent_name, orig_policy_name, alt_policy_name):
    feature_names = feature_schema['feature_names']
    n_features = len(feature_names)

    exp = Explanation(n_features, feature_names, agent_name, orig_policy_name, alt_policy_name)

    for i in range(n_features):
        feature_orig_vals = [outcome_orig[i].item() for outcome_orig, outcome_alt in outcome_pairs]
        feature_alt_vals = [outcome_alt[i].item() for outcome_orig, outcome_alt in outcome_pairs]
        if feature_names[i].startswith('abs'):
            feature_orig_vals = [abs(outcome_orig[i].item()) for outcome_orig, outcome_alt in outcome_pairs]
            feature_alt_vals = [abs(outcome_alt[i].item()) for outcome_orig, outcome_alt in outcome_pairs]

        # see if feature in alternative scenario is larger or smaller on average than in original
        mean_diff = np.mean(feature_orig_vals) - np.mean(feature_alt_vals)
        if mean_diff > 0:
            exp.set_rel(feature_names[i], 'larger')
        elif mean_diff < 0:
            exp.set_rel(feature_names[i], 'smaller')

        # see if difference is significant
        _, p_val = stats.ttest_rel(feature_orig_vals, feature_alt_vals)
        exp.set_p_val(feature_names[i], p_val)

    return exp


class Outcome:
     def __init__(self, end_state, end_state_val):
         self.end_state = end_state
         self.end_state_val = end_state_val


class Explanation:
    def __init__(self, n_features, feature_names, agent_name, orig_policy_name, alt_policy_name):
        self.feature_names = feature_names
        self.agent_name = agent_name
        self.orig_policy_name = orig_policy_name
        self.alt_policy_name = alt_policy_name
        self.p_vals = {f_name: 0 for f_name in feature_names}  # holds p-value for each feature
        self.rel = {f_name: 'equal' for f_name in feature_names}  # stores the relationships of feature to the alternative scenario
        self.thres = 0.05

    def get_rel_encoded(self):
        encoding = {'equal': 0,
                    'larger': 1,
                    'smaller': -1}
        rel_encoded = {f_name: encoding[rel] for f_name, rel in self.rel.items()}
        return rel_encoded

    def set_p_val(self, feature_name, p_val):
        self.p_vals[feature_name] = p_val

    def set_rel(self, feature_name, rel):
        self.rel[feature_name] = rel

    def set_state_val_importance(self, val):
        self.state_val_importance = val

    def print(self):
        s = 'Agent A prefers states with: \n'

        for feature in self.feature_names:
            if (self.rel[feature] != 'equal') and (self.p_vals[feature] < self.thres):
                s += '\n \t{} {}'.format(feature, self.rel[feature])

        print(s)