from dsl import *
from env_settings import *

import numpy as np
import pickle 


# with open('plps.pkl', 'rb') as f: 
#     plps,plp_priors = pickle.load(f)


class StateActionProgram(object):
    """
    A callable object with input (state, action) and Boolean output.

    Made a class to have nice strs and pickling and to avoid redundant evals.
    """
    def __init__(self, program):
        self.program = program
        self.wrapped = None

    def __call__(self, *args, **kwargs):
        # if self.wrapped is None:
        self.wrapped = eval('lambda s, loc, a: ' + self.program)
        return self.wrapped(*args, **kwargs)

    def __repr__(self):
        return self.program

    def __str__(self):
        return self.program

    def __getstate__(self):
        return self.program

    def __setstate__(self, program):
        self.program = program
        self.wrapped = None

    def __add__(self, s):
        if isinstance(s, str):
            return StateActionProgram(self.program + s)
        elif isinstance(s, StateActionProgram):
            return StateActionProgram(self.program + s.program)
        raise Exception()

    def __radd__(self, s):
        if isinstance(s, str):
            return StateActionProgram(s + self.program)
        elif isinstance(s, StateActionProgram):
            return StateActionProgram(s.program + self.program)
        raise Exception()

class PLPPolicy(object):
    #Debug
    # with open("test.pkl", "rb") as f:
    #      plp_t, obs_t, action_and_loc_t = pickle.load(f)

    def __init__(self, plps, probs, seed=0, map_choices=True):
        assert abs(np.sum(probs) - 1.) < 1e-5

        self.plps = plps
        self.probs = probs
        self.map_choices = map_choices
        self.rng = np.random.RandomState(seed)

        self._action_prob_cache = {}

    def __call__(self, obs):
        #No policy found
        if isinstance(self.get_action_probs(obs), int) and self.get_action_probs(obs) == -1: 
            return -1 
        action_probs = self.get_action_probs(obs).flatten()
        if self.map_choices:
            idx = np.argmax(action_probs).squeeze()
        else:
            idx = self.rng.choice(len(action_probs), p=action_probs)
        return np.unravel_index(idx, obs.shape + (4,))

    def hash_obs(self, obs):
        return tuple(tuple(l) for l in obs)
    
    def action_conv(self,a):
        b = {'pass':0, 'x':1, 'y':2,'z':3}
        return b[a]

    def get_action_probs(self, obs):
        hashed_obs = self.hash_obs(obs)
        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]

        action_probs = np.zeros((obs.shape + (4,)), dtype=np.float32)

        for plp, prob in zip(self.plps, self.probs):
            for r, c, a in self.get_plp_suggestions(plp, obs):
                a = self.action_conv(a)
                action_probs[r, c, a] += prob

        denom = np.sum(action_probs)
        if denom == 0.:
            action_probs = -1  # 1./(action_probs.shape[0] * action_probs.shape[1] * action_probs.shape[2])
        else:
            action_probs = action_probs / denom
        self._action_prob_cache[hashed_obs] = action_probs
        return action_probs

    def get_plp_suggestions(self, plp, obs):
        suggestions = []

        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                for a in xyz.ALL_ACTION_TOKENS: #('xyz.PASS','xyz.X','xyz.Y','xyz.Z'):
                        if plp(obs, (r,c), a):
                            suggestions.append((r, c, a))

        return suggestions
