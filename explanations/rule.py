class Rule:

    def __init__(self, feature_schema):
        self.feature_schema = feature_schema
        self.feature_thres = {}

    def update_thresholds(self, threshold_dict):
        for feature_name, val in threshold_dict.items():
            self.feature_thres[feature_name] = val

    def is_supported(self, x, policy_id):
        # checks whether instance x is supported by the rule
        if len(x.keys()) != len(self.feature_thres.keys()):
            return False

        outcome = self.feature_schema['target']
        features = self.feature_schema['features']

        for f, val in x.items():
            if f not in self.feature_thres.keys():
                return False  # x has features that rule doesn't cover

            if f in features:
                thres_min, thres_max = self.feature_thres[f]
                if val < thres_min or val > thres_max:
                    return False

            if f == outcome:
                rule_outcome = self.feature_thres[outcome]
                action = rule_outcome.split('-')[policy_id]
                action = int(action)

                if action != val:  # TODO: works only for cancer - need to add step for inv task
                    return False

        return True

    def print(self):
        feature_names = self.feature_schema['features']
        outcome_name = self.feature_schema['target']

        s = ''
        s += '{}: {} '.format(outcome_name, self.feature_thres[outcome_name])

        for f_n in feature_names:
            val = self.feature_thres[f_n]
            s += '{}: [{}, {}] '.format(f_n, val[0], val[1])

        print(s)
