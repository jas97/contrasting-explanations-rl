import json
import numpy as np
import heapq

from autorl4do.explanations.util import append_to_json_file


def extract_rules(tree, feature_schema, file, policy):
    tree_ = tree.tree_

    children_left = tree_.children_left
    children_right = tree_.children_right
    split_vars = tree_.feature
    thresholds = tree_.threshold
    values = tree_.value
    classes = tree.classes_

    node_chains = get_paths(0, None, None, children_left, children_right)

    outcome_name = feature_schema['target']

    rules = []
    for chain in node_chains:
        r = {}  # curr rule
        # extract leaf value
        pred = values[chain[-1]]
        pred = np.argmax(pred)  # TODO: adjust for regression to be the above line
        tree_class = classes[pred.item()]
        r[outcome_name] = str(tree_class)
        r = init_rule(r, feature_schema, policy)

        for i, n in enumerate(chain[:-1]):
            child_left = children_left[n]
            child_right = children_right[n]

            if chain[i + 1] == child_left:  # taken left turn between nodes
                direction = -1
            elif chain[i + 1] == child_right:  # taken right turn between nodes
                direction = 1

            r = update_rule(r, split_vars[n], thresholds[n], direction, feature_schema)
        rules.append(r)

    append_to_json_file(file, rules)

    return rules


def init_rule(r, feature_schema, policy):
    lows = feature_schema['low']
    highs = feature_schema['high']

    r['policy'] = policy

    # limits for state features are env limits
    for i, f in enumerate(feature_schema['features']):
        start = round(np.float64(lows[i]), 4)
        end = round(np.float64(highs[i]), 4)
        r[f] = [start, end]

    return r


def get_paths(root, paths, curr_path, children_left, children_right):
    # returns all path through decision tree from root to leaf
    if paths is None:
        paths = []
    if curr_path is None:
        curr_path = []

    curr_path.append(root)
    if children_left[root] == children_right[root]:
        paths.append(curr_path)
    else:
        children = [children_left[root], children_right[root]]
        for child in children:
            get_paths(child, paths, list(curr_path), children_left, children_right)

    return paths


def update_rule(rule, var, thres, direction, feature_schema):
    lows = feature_schema['low']
    highs = feature_schema['high']
    policy_key = feature_schema['policy_key']

    var_key = str(var)

    thres = round(thres, 4)

    if var_key == str(policy_key):
        if direction == -1:
            rule['policy'] = 'A'
        else:
            rule['policy'] = 'B'
        return rule

    if var_key in rule.keys():
        low, high = rule[var_key]
        if direction == -1:
            rule[var_key] = [np.float64(low), np.float64(thres)]
        else:
            rule[var_key] = [np.float64(thres), np.float64(high)]
    else:
        if direction == -1:
            rule[var_key] = [np.float64(lows[var]), np.float64(thres)]
        else:
            rule[var_key] = [np.float64(thres), np.float64(highs[var])]

    return rule


def generate_contrastive_strategies(rule_set, df, feature_schema, save_path, max_rules=15):
    strategies = []

    for rule in rule_set:
        contr_rule = get_contrasting_rule(rule, rule_set, feature_schema)
        strategies.append((rule, contr_rule))

    # best_strategies = get_best_contrastive_strategies(strategies, df, feature_schema, max_rules=max_rules)

    # append_to_json_file(save_path, best_strategies)

    return strategies


def get_contrasting_rule(rule, rule_set, feature_schema):
    policy = feature_schema['policy']
    features = feature_schema['features']

    contr_rules = []
    contr_rules_sim = []

    for alt_rule in rule_set:
        if alt_rule[policy] != rule[policy]:  # contrastive rule should be about the alternative policy
            # calculate similarity between alt_rule and rule
            sim = get_similarity_for_rules(rule, alt_rule, features)

            # TODO: maybe divide sim with num features to keep it [0, 1]
            contr_rules.append(alt_rule)
            contr_rules_sim.append(sim)

    # select alternative rule with biggest similarity score
    max_index = np.argmax(np.array(contr_rules_sim))
    best_alt_rule = contr_rules[max_index]

    return best_alt_rule


def load_rules(file):
    with open(file, 'r') as f:
        rules = json.load(f)

        return rules


def get_similarity_for_rules(r1, r2, features):
    sim = 0.0
    for f in features:
        rangeA = r1[f]  # get thresholds for this features in both rules
        rangeB = r2[f]
        sim += get_similarity_for_ranges(rangeA, rangeB)

    return sim


def get_similarity_for_ranges(rangeA, rangeB):
    # calculates similarity between 2 ranges
    first = rangeA if rangeA[0] <= rangeB[0] else rangeB  # interval that has smaller start value
    second = rangeB if first == rangeA else rangeA

    if first[1] < second[0]:
        return 0  # no overlap

    if first[0] == second[0] and first[1] == second[1]:
        return 1  # intervals are completely overlapping

    if first[1] < second[1]:  # overlap
        return abs(first[1] - second[0])*1.0 / abs(first[0] - second[1])

    return abs(second[0] - second[1])*1.0 / abs(first[0] - first[1])  # second is contained in first


def get_best_contrastive_strategies(rule_pairs, df, feature_schema, max_rules=10):
    features = feature_schema['features']
    scores = []

    for r1, r2 in rule_pairs:
        sim = get_similarity_for_rules(r1, r2, features)
        support = get_rule_support(r1, df, feature_schema) + get_rule_support(r2, df, feature_schema)

        scores.append((sim + support) / 3)

    if len(scores) < max_rules:
        max_rules = len(scores)

    max_ind = heapq.nlargest(max_rules, range(len(scores)), scores.__getitem__)
    best_rules = []
    for i, (r1, r2) in enumerate(rule_pairs):
        if i in max_ind:
            best_rules.append((r1, r2))

    return best_rules


def get_rule_support(r, df, feature_schema):
    count = 0.0

    policy = feature_schema['policy']
    features = feature_schema['features']

    for index, row in df.iterrows():
        supports = True
        if row[policy] != 'any' and row[policy] != r[policy]:
            supports = True

        for f in features:
            low, high = r[f]
            feature_val = row[f]

            if feature_val < low or feature_val > high:
                supports = False

        if supports:
            count += 1

    support = count / df.shape[0]
    return support







