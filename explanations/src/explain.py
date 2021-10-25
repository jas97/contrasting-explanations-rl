from explanations.src.contrastive_explanations import get_contrastive_explanations
from explanations.src.policy_comparison import get_pref_trajectories, build_tree, get_disagreement_data
from explanations.src.rule_extraction import extract_rules, load_rules, generate_contrastive_strategies


class ExplainGen():

    def __init__(self, task, feature_names):
        self.task = task
        self.feature_names = feature_names
        self.num_features = len(self.feature_names)

        self.results_path = 'results/{}/tree'.format(self.task)
        self.rules_path = 'rules/{}.json'.format(self.task)
        self.contr_rules_path = 'rules/{}_contr.json'.format(self.task)
        self.df_path = 'results/{}/df.csv'.format(self.task)

    def explain(self, env, policy_A, policy_B, max_traj_len=10, num_episodes=100, step=None, exp_type='all'):
        print('Generating explanations type = {}'.format(exp_type))

        disagreement_states, pref_trajectories, outcomes = get_pref_trajectories(policy_A,
                                                                                 policy_B,
                                                                                 env,
                                                                                 max_traj_len=max_traj_len,
                                                                                 num_episodes=num_episodes)



        if exp_type == 'all' or exp_type == 'strategies':
            df, feature_schema = get_disagreement_data(pref_trajectories, policy_A, policy_B, env, step=step)
            df.to_csv(self.df_path)

            feature_schema['low'] = [round(min(df[col]), 4) for col in feature_schema['features']]
            feature_schema['high'] = [round(max(df[col]), 4) for col in feature_schema['features']]
            feature_schema['feature_names'] = self.feature_names

            treeA = build_tree(df[df.policy == 0], feature_schema, self.results_path + '_A', self.feature_names)
            treeB = build_tree(df[df.policy == 1], feature_schema, self.results_path + '_B', self.feature_names)

            extract_rules(treeA, feature_schema, file=self.rules_path, policy=0)
            extract_rules(treeB, feature_schema, file=self.rules_path, policy=1)

            rule_set = load_rules(self.rules_path)

            print('Generating contrastive strategies:')
            contr_strategies = generate_contrastive_strategies(rule_set, df, feature_schema, self.contr_rules_path)
            for s1, s2 in contr_strategies:
                print(s1)
                print(s2)
                print('-' * 80)

        if exp_type == 'all' or exp_type == 'explanations':
            exp = get_contrastive_explanations(outcomes,
                                               env,
                                               feature_schema={'feature_names': self.feature_names},
                                               agent_names=['A', 'B'],
                                               policy_names=['A', 'B'])
            exp.print()

        return len(disagreement_states) if disagreement_states is not None else 0





