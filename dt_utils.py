from policy import StateActionProgram

import numpy as np


def get_path_to_leaf(leaf, parents):
    reverse_path = []
    parent, parent_choice = parents[leaf]

    while True:
        reverse_path.append((parent, parent_choice))
        if parents[parent] is None:
            break
        parent, parent_choice = parents[parent]

    return reverse_path[::-1]

def get_conjunctive_program(path, node_to_features, features, feature_log_probs):
    program = '('
    log_p = 0.

    for i, (node_id, sign) in enumerate(path):
        feature_idx = node_to_features[node_id]
        precondition = features[feature_idx]
        feature_log_p = feature_log_probs[feature_idx]

        if sign == 'right':
            log_p += feature_log_p
        else:
            assert sign == 'left'
            #log_p += feature_log_p
            if log_p == 0:
                log_p = np.log((1-np.exp(feature_log_p)))
            else:
                log_p += np.log((1-np.exp(feature_log_p))) 
            


        if sign == 'right':
            program = program + precondition
        else:
            assert sign == 'left'
            program = program + 'not (' + precondition + ')'

        if i < len(path) - 1:
            program = program + ' and '

    program = program + ')'

    return program, log_p

def get_disjunctive_program(conjunctive_programs):
    if len(conjunctive_programs) == 0:
        return 'False'

    program = ''

    for i, conjunctive_program in enumerate(conjunctive_programs):
        program = program +'(' + conjunctive_program + ')'
        if i < len(conjunctive_programs) - 1:
            program = program + ' or '

    return program

def extract_plp_from_dt(clf, features, feature_log_probs, num_positive_demo):
    estimator = clf

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    node_to_features = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    value = estimator.tree_.value.squeeze()

    stack = [0]
    parents = {0 : None}
    true_leaves = []
    true_leaves_dict = dict()
    total_pos_leaf = 0
    total_neg_leaf = 0
    variance = 0
    try:
        while len(stack) > 0:
            node_id = stack.pop()

            if (children_left[node_id] != children_right[node_id]):
                assert 0 < threshold[node_id] < 1
                stack.append(children_left[node_id])
                parents[children_left[node_id]] = (node_id, 'left')
                stack.append(children_right[node_id])
                parents[children_right[node_id]] = (node_id, 'right')
            elif value[node_id][1] > value[node_id][0]: #  != 0: #
                total_pos_leaf = value[node_id][1] + total_pos_leaf
                total_neg_leaf = value[node_id][0] + total_neg_leaf
                true_leaves_dict[node_id] = np.log(value[node_id][1]/(value[node_id][0] +value[node_id][1]))
                true_leaves.append(node_id)

        print("Sum of total positive leaves {}".format(total_pos_leaf))
        print("Sum of total negative leaves {}".format(total_neg_leaf))
        print("Toal number of leaf nodes {}".format(len(true_leaves)))
        # try:
        #     print("Likelihood? {}".format(np.log(total_pos_leaf/(total_pos_leaf+total_neg_leaf))))
        #     likelihood = np.log(total_pos_leaf/(total_pos_leaf+total_neg_leaf))
        # except:
        #     print("Likelihood {}".format("Nan"))
        #     likelihood = float("-inf")

        # if total_pos_leaf != num_positive_demo:
        #     print("Likelihood {}".format("Nan"))
        #     likelihood = float("-inf")
        # else:
        likelihood = np.log(total_pos_leaf/(total_pos_leaf+total_neg_leaf))

        min_likelihood = 0
        for i in true_leaves_dict.keys():
            min_likelihood = min(min_likelihood,true_leaves_dict[i])
            variance = (true_leaves_dict[i] - likelihood)**2+variance
        #print("Min Likelihood: {}".format(min_likelihood))
        

        paths_to_true_leaves = [get_path_to_leaf(leaf, parents) for leaf in true_leaves]

        conjunctive_programs = []
        program_log_prob = 0

        for path in paths_to_true_leaves:
            and_program, log_p = get_conjunctive_program(path, node_to_features, features, feature_log_probs)
            conjunctive_programs.append(and_program)
            if program_log_prob == 0:
                program_log_prob = log_p
            else:
                program_log_prob = np.log(np.exp(log_p)+np.exp(program_log_prob))


        disjunctive_program = get_disjunctive_program(conjunctive_programs)

        if not isinstance(disjunctive_program, StateActionProgram):
            disjunctive_program = StateActionProgram(disjunctive_program)
        print("Likelihood: {}".format(program_log_prob))
    except:
        return None, -1, float("-inf"), 0, float("-inf")
    return disjunctive_program, program_log_prob, likelihood, len(true_leaves), min_likelihood



