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
    def __init__(self, program, base_class = "Hello"):
        self.base_class = base_class
        self.program = program
        self.wrapped = None

    def __call__(self, *args, **kwargs):
        #if self.wrapped is None:
        if self.base_class == "PlayingWithXYZ":
            self.wrapped = eval('lambda s, loc, a: ' + self.program)
        else:
            self.wrapped = eval('lambda s, loc: ' + self.program)
        return self.wrapped(*args, **kwargs)

    def __repr__(self):
        return self.program

    def __str__(self):
        return self.program

    def __getstate__(self):
        fin = dict()
        fin['base_class'] = self.base_class 
        fin['program'] = self.program
        return fin

    def __setstate__(self, fin):
        self.program = fin['program']
        self.wrapped = None
        self.base_class = fin['base_class']

    def __add__(self, s):
        if isinstance(s, str):
            return StateActionProgram(self.program + s, self.base_class )
        elif isinstance(s, StateActionProgram):
            return StateActionProgram(self.program + s.program, self.base_class )
        raise Exception()

    def __radd__(self, s):
        if isinstance(s, str):
            return StateActionProgram(s + self.program, self.base_class )
        elif isinstance(s, StateActionProgram):
            return StateActionProgram(s.program + self.program, self.base_class )
        raise Exception()

class PLPPolicy(object):
    #Debug
    # with open("test.pkl", "rb") as f:
    #      plp_t, obs_t, action_and_loc_t = pickle.load(f)

    def __init__(self, plps, probs, seed=0, map_choices=True,base_class=None):
        assert abs(np.sum(probs) - 1.) < 1e-5

        self.plps = plps
        self.probs = probs
        self.map_choices = map_choices
        self.rng = np.random.RandomState(seed)
        self.base_class = base_class

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
        
        if self.base_class == "PlayingWithXYZ":
            return np.unravel_index(idx, obs.shape + (len(xyz.ALL_ACTION_TOKENS),))
        else:
            return np.unravel_index(idx, obs.shape)


    def hash_obs(self, obs):
        return tuple(tuple(l) for l in obs)
    
    def action_conv(self,a):
        b = {'pass':0, 'x':1, 'y':2,'z':3, 'empty':4}
        return b[a]

    def get_action_probs(self, obs):
        hashed_obs = self.hash_obs(obs)
        if hashed_obs in self._action_prob_cache:
            return self._action_prob_cache[hashed_obs]


        if self.base_class == "PlayingWithXYZ":
            action_probs = np.zeros((obs.shape + (len(xyz.ALL_ACTION_TOKENS),)), dtype=np.float32)
            for plp, prob in zip(self.plps, self.probs):
                for r, c, a in self.get_plp_suggestions(plp, obs):
                    a = self.action_conv(a)
                    action_probs[r, c, a] += prob
        else:
            action_probs = np.zeros(obs.shape, dtype=np.float32)
            for plp, prob in zip(self.plps, self.probs):
                for r, c in self.get_plp_suggestions(plp, obs):
                    action_probs[r, c] += prob

        

        denom = np.sum(action_probs)
        if denom == 0.:
            action_probs = -1  # 1./(action_probs.shape[0] * action_probs.shape[1] * action_probs.shape[2])
        else:
            action_probs = action_probs / denom
        self._action_prob_cache[hashed_obs] = action_probs
        return action_probs

    def get_plp_suggestions(self, plp, obs):
        suggestions = []

        if self.base_class == "PlayingWithXYZ":
            for r in range(obs.shape[0]):
                for c in range(obs.shape[1]):
                    for a in xyz.ALL_ACTION_TOKENS: #('xyz.PASS','xyz.X','xyz.Y','xyz.Z'):
                            if plp(obs, (r,c), a):
                                suggestions.append((r, c, a))
        else:
            for r in range(obs.shape[0]):
                for c in range(obs.shape[1]):
                    if plp(obs, (r,c)):
                        suggestions.append((r, c))


        return suggestions
