import numpy as np


### Methods
def out_of_bounds(r, c, shape):
    return (r < 0 or c < 0 or r >= shape[0] or c >= shape[1])

def shifted(direction, local_program, cell, obs):
    if cell is None:
        new_cell = None
    else:
        new_cell = (cell[0] + direction[0], cell[1] + direction[1])
    return local_program(new_cell, obs)

def cell_is_value(value, cell, obs):
    if cell is None or out_of_bounds(cell[0], cell[1], obs.shape):
        focus = None
    else:
        focus = obs[cell[0], cell[1]]

    return (focus == value)

def at_cell_with_value(value, local_program, obs):
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
        return False
    else:
        cell = matches[0]
    return local_program(cell, obs)

def at_all_cell_with_value(value, local_program, obs):
    matches = np.argwhere(obs == value)
    if len(matches) == 0:
        cell = None
        return False#local_program(cell,obs)
    curr_cond = True
    for match in matches:
        curr_cond = curr_cond and local_program(match,obs)
    return curr_cond

def at_action_cell(local_program, cell, obs):
    return local_program(cell, obs)

def is_action(state, loc, action, value):
    if action == value:
        return True
    else: return False

def is_there_on(direction,value, cell ,obs):
    if np.any(cell == None):
        return False
    if direction == "NORD":
        return np.any(obs[(cell[0]+1):,...]==value)

    if direction == "SOUTH":
        return np.any(obs[:cell[0],...]==value)

    if direction == "EAST":
        return np.any(obs[...,(cell[0]+1):]==value)
        
    if direction == "WEST":
        return np.any(obs[:cell[0],...]==value)
    

def is_around(radious,value,cell,obs):
    return np.any(n_closest(obs,cell,d=radious)==value)

def n_closest(x,n,d=1):
    return x[n[0]-d:n[0]+d+1,n[1]-d:n[1]+d+1]

def is_finish(loc):
    if loc == (0,0):
        return True
    else:
        return False

def scanning(direction, true_condition, false_condition, cell, obs, max_timeout=1):
    if cell is None:
        return False

    for _ in range(max_timeout):
        cell = (cell[0] + direction[0], cell[1] + direction[1])

        if true_condition(cell, obs):
            return True

        if false_condition(cell, obs):
            return False

        # prevent infinite loops
        if out_of_bounds(cell[0], cell[1], obs.shape):
            return False

    return False



### Grammatical Prior
#START, LOCAL_ACTION_PROGRAM,LOCAL_STATE_PROGRAM,CONDITION,DIRECTION,POSITIVE_NUM, NEGATIVE_NUM, CARDINAL_DIR, VALUE = range(9)
START, LOCAL_ACTION_PROGRAM,LOCAL_STATE_PROGRAM,CONDITION,RADIOUS, CARDINAL_DIR, VALUE = range(7)


def create_grammar_unity(object_types):
    grammar = {
        START : ([['is_finish(loc)'],
                  ['at_cell_with_value(', VALUE, ',', LOCAL_STATE_PROGRAM, ', s)'],
                  ['at_action_cell(', LOCAL_ACTION_PROGRAM, ', loc, s)'],
                  ['at_all_cell_with_value(',VALUE,',',LOCAL_STATE_PROGRAM,', s)']],
                  [0.25,0.25, 0.25,0.25]),
        LOCAL_ACTION_PROGRAM : ([['lambda cell,o : is_there_on(',CARDINAL_DIR,',', VALUE, ', cell, o)'],
                          ['lambda cell,o : shifted( (0,0)', ',', CONDITION, ', cell, o)'],
                          ['lambda cell,o : is_around(',RADIOUS,',', VALUE, ', cell, o)']],
                          [0.33,0.33,0.33]),
        LOCAL_STATE_PROGRAM : ([['lambda cell,o : is_there_on(', CARDINAL_DIR,',', VALUE, ', cell, o)'],
                                ['lambda cell,o : is_around(',RADIOUS,',', VALUE, ', cell, o)']],
                          [0.5,0.5]),
        CONDITION : ([['lambda cell,o : cell_is_value(', VALUE, ', cell, o)'],],
                      #['lambda cell,o : scanning(', DIRECTION, ',', LOCAL_PROGRAM, ',', LOCAL_PROGRAM, ', cell, o)']],
                      [1]),
        # DIRECTION : ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
        #                ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
        #                ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
        #                ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
        #               [1./8] * 8),
        # POSITIVE_NUM : ([['1']], #[POSITIVE_NUM, '+1']],
        #                  [1]),
        # NEGATIVE_NUM : ([['-1']],#, [NEGATIVE_NUM, '-1']],
        #                  [1]),
        RADIOUS: ([['1'],['5'],['10']],[0.2,0.5,0.7]),
        CARDINAL_DIR : ([["'NORD'"],["'SOUTH'"],["'WEST'"],["'EAST'"]],[0.25,0.25,0.25,0.25]),
        VALUE : (tuple(obj for obj in object_types if obj != "None" and obj != "unity.START" and obj != "unity.CLICK" and obj != 'unity.PASS'), 
                 [1./len(object_types) for _ in object_types if _ != "None"])
    }
    return grammar

def create_grammar(object_types):
    grammar = {
        START : ([['at_cell_with_value(', VALUE, ',', LOCAL_PROGRAM, ', s)'],
                ['at_action_cell(', LOCAL_PROGRAM, ', loc, s)']],
                [0.5, 0.5]),
        LOCAL_PROGRAM : ([[CONDITION],
                        ['lambda cell,o : shifted(', DIRECTION, ',', CONDITION, ', cell, o)']],
                        [0.5, 0.5]),
        CONDITION : ([['lambda cell,o : cell_is_value(', VALUE, ', cell, o)'],],
                    #['lambda cell,o : scanning(', DIRECTION, ',', LOCAL_PROGRAM, ',', LOCAL_PROGRAM, ', cell, o)']],
                    [1]),
        DIRECTION : ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
                    ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
                    ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
                    ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
                    [1./8] * 8),
        POSITIVE_NUM : ([['1'], [POSITIVE_NUM, '+1']],
                        [0.5, 0.5]),
        NEGATIVE_NUM : ([['-1'], [NEGATIVE_NUM, '-1']],
                        [0.5, 0.5]),
        VALUE : (object_types, 
                [1./len(object_types) for _ in object_types])
    }
    return grammar

# def create_grammar(object_types):
#     grammar = {
#         START : ([['at_cell_with_value(', VALUE, ',', LOCAL_PROGRAM, ', s)'],
#                   ['at_action_cell(', LOCAL_PROGRAM, ', a, s)']],
#                   [0.5, 0.5]),
#         LOCAL_PROGRAM : ([[CONDITION],
#                           ['lambda cell,o : shifted(', DIRECTION, ',', CONDITION, ', cell, o)']],
#                           [0.5, 0.5]),
#         CONDITION : ([['lambda cell,o : cell_is_value(', VALUE, ', cell, o)'],
#                       ['lambda cell,o : scanning(', DIRECTION, ',', LOCAL_PROGRAM, ',', LOCAL_PROGRAM, ', cell, o)']],
#                       [0.5, 0.5]),
#         DIRECTION : ([['(', POSITIVE_NUM, ', 0)'], ['(0,', POSITIVE_NUM, ')'],
#                       ['(', NEGATIVE_NUM, ', 0)'], ['(0,', NEGATIVE_NUM, ')'],
#                       ['(', POSITIVE_NUM, ',', POSITIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', POSITIVE_NUM, ')'],
#                       ['(', POSITIVE_NUM, ',', NEGATIVE_NUM, ')'], ['(', NEGATIVE_NUM, ',', NEGATIVE_NUM, ')']],
#                      [1./8] * 8),
#         POSITIVE_NUM : ([['1'], [POSITIVE_NUM, '+1']],
#                          [0.99, 0.01]),
#         NEGATIVE_NUM : ([['-1'], [NEGATIVE_NUM, '-1']],
#                          [0.99, 0.01]),
#         VALUE : (object_types, 
#                  [1./len(object_types) for _ in object_types])
#     }
#     return grammar
    