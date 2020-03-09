from cache_utils import manage_cache
from dsl import *
from env_settings import *
from grammar_utils import generate_programs, generate_programs_test
from dt_utils import extract_plp_from_dt
from expert_demonstrations import get_demonstrations, get_interactive_demo
from policy import StateActionProgram, PLPPolicy
from utils import run_single_episode

from collections import defaultdict
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, lil_matrix, vstack
from copy import deepcopy
import gym
import multiprocessing
import numpy as np
import time
import os
import matplotlib
matplotlib.use('TkAgg')

cache_dir = 'cache'


@manage_cache(cache_dir, ['.pkl', '.pkl'])
def get_program_set(base_class_name, num_programs):
    """
    Enumerate all programs up to a certain iteration.

    Parameters
    ----------
    base_class_name : str
    num_programs : int

    Returns
    -------
    programs : [ StateActionProgram ]
        A list of programs in enumeration order.
    program_prior_log_probs : [ float ]
        Log probabilities for each program.
    """
    object_types = get_object_types(base_class_name)
    grammar = create_grammar(object_types)

    # if base_class_name=="PlayingWithXYZ":
    #     program_generator = generate_programs_test(grammar)

    program_generator = generate_programs_test(grammar)
    programs = []
    program_prior_log_probs = []

    print("Generating {} programs".format(num_programs))
    for _ in range(num_programs):
        program, lp = next(program_generator)
        programs.append(program)
        program_prior_log_probs.append(lp)
    print("\nDone.")

    return programs, program_prior_log_probs

def extract_examples_from_demonstration_item(demonstration_item):
    """
    Convert a demonstrated (state, action) into positive and negative classification data.

    All actions not taken in the demonstration_item are considered negative.

    Parameters
    ----------
    demonstrations : (np.ndarray, (int, int))
        A state, action pair.

    Returns
    -------
    positive_examples : [(np.ndarray, (int, int))]
        A list with just the input state, action pair (for convenience).
    negative_examples : [(np.ndarray, (int, int))]
        A list with negative examples of state, actions.
    """
    state, action = demonstration_item

    positive_examples = [(state, action)]
    negative_examples = []

    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            if (r, c) == action:
                continue
            else:
                negative_examples.append((state, (r, c)))

    return positive_examples, negative_examples

def extract_examples_from_demonstration(demonstration):
    """
    Convert demonstrated (state, action)s into positive and negative classification data.

    Parameters
    ----------
    demonstrations : [(np.ndarray, (int, int))]
        State, action pairs

    Returns
    -------
    positive_examples : [(np.ndarray, (int, int))]
        A list with just the input state, action pairs (for convenience).
    negative_examples : [(np.ndarray, (int, int))]
        A list with negative examples of state, actions.
    """
    positive_examples = []
    negative_examples = []

    for demonstration_item in demonstration:
        demo_positive_examples, demo_negative_examples = extract_examples_from_demonstration_item(demonstration_item)
        positive_examples.extend(demo_positive_examples)
        negative_examples.extend(demo_negative_examples)

    return positive_examples, negative_examples


class PlayingWithXYZ:
    object_types = ('pass','x','y','z')    
    
    @classmethod
    def extract_examples_from_demonstration(cls,demonstration):
        positive_examples = []
        negative_examples = []

        for demonstration_item in demonstration:
            demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item(demonstration_item)
            positive_examples.extend(demo_positive_examples)
            negative_examples.extend(demo_negative_examples)

        return positive_examples, negative_examples

    @classmethod
    def extract_examples_from_demonstration_item(cls,demonstration_item):
        state, loc_and_action = demonstration_item
        a, loc = loc_and_action

        positive_examples = [(state, loc, a)]
        negative_examples = []

        for r in range(state.shape[0]):
            for c in range(state.shape[1]):
                for val in cls.object_types:
                    if val == a and (r, c) == loc:
                        continue
                    else:
                        negative_examples.append((state, (r, c), val))

        return positive_examples, negative_examples

    # @classmethod
    # def retrofit_demonstrations(cls,fn_inputs):
    #     acton_inputs = deepcopy(fn_inputs)
    #     for idx in range(len(fn_inputs)):
    #         fn_inputs[idx] = list(fn_inputs[idx])
    #         fn_inputs[idx][1] = fn_inputs[idx][1][1]
    #         fn_inputs[idx] = tuple(fn_inputs[idx])

    #         acton_inputs[idx] = list(acton_inputs[idx])
    #         acton_inputs[idx][1] = acton_inputs[idx][1][0]
    #         acton_inputs[idx] = tuple(acton_inputs[idx])
    #     return fn_inputs, acton_inputs
    
def apply_programs(programs, fn_input):
    """
    Worker function that applies a list of programs to a single given input.

    Parameters
    ----------
    programs : [ callable ]
    fn_input : Any

    Returns
    -------
    results : [ bool ]
        Program outputs in order.
    """
    x = []
    for program in programs:
        try:
            x_i = program(*fn_input)
        except: 
            print(program)
            print(fn_input)
            x_i = program(*fn_input)
            exit()
        x.append(x_i)
    return x

@manage_cache(cache_dir, ['.npz', '.pkl'])
def run_all_programs_on_single_demonstration(base_class_name, num_programs, demo_number, program_interval=1000, interactive=False):
    """
    un all programs up to some iteration on one demonstration.

    Expensive in general because programs can be slow and numerous, so caching can be very helpful.

    Parallelization is designed to save time in the regime of many programs.

    Care is taken to avoid memory issues, which are a serious problem when num_programs exceeds 50,000.

    Returns classification dataset X, y.

    Parameters
    ----------
    base_class_name : str
    num_programs : int
    demo_number : int
    program_interval : int
        This interval splits up program batches for parallelization.

    Returns
    -------
    X : csr_matrix
        X.shape = (num_demo_items, num_programs)
    y : [ bool ]
        y.shape = (num_demo_items,)
    """

    print("Running all programs on {}, {}".format(base_class_name, demo_number))

    programs, _ = get_program_set(base_class_name, num_programs)
    

    #Max demo needed because a demostration is considered as well doing nothing. Hence legth is set
    if base_class_name == "PlayingWithXYZ":
        demonstration = get_demonstrations(base_class_name, demo_numbers=(demo_number,),  max_demo_length=1, interactive=interactive)
        positive_examples, negative_examples  = PlayingWithXYZ.extract_examples_from_demonstration(demonstration)
    else: 
        demonstration = get_demonstrations(base_class_name, demo_numbers=(demo_number,))
        positive_examples, negative_examples = extract_examples_from_demonstration(demonstration)
    
    y = [1] * len(positive_examples) + [0] * len(negative_examples)

    num_data = len(y)
    num_programs = len(programs)

    X = lil_matrix((num_data, num_programs), dtype=bool)

    fn_inputs = positive_examples + negative_examples
    l = 0
    # if base_class_name == "PlayingWithXYZ":
    #     l = 4 
    #     # fn_inputs, action_fn_inputs = PlayingWithXYZ.retrofit_demonstrations(fn_inputs)
    #     for i in range(l):
    #         print('Iteration {} of {}'.format(i, num_programs), end='\r')
    #         end = min(i+program_interval, num_programs)
    #         fn = partial(apply_programs, [programs[i]])
    #         results = map(fn, action_fn_inputs)
    #         for X_idx, x in enumerate(results):
    #             X[X_idx,i] = x 

    # This loop avoids memory issues
    for i in range(l, num_programs, program_interval):
        end = min(i+program_interval, num_programs)
        print('Iteration {} of {}'.format(i, num_programs), end='\r')

        num_workers = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(num_workers)

        fn = partial(apply_programs, programs[i:end])
        
        results = pool.map(fn, fn_inputs)
        pool.close()

        for X_idx, x in enumerate(results):
            X[X_idx, i:end] = x

    X = X.tocsr()

    print()
    return X, y

def run_all_programs_on_demonstrations(base_class_name, num_programs, demo_numbers, interactive=False):
    """
    See run_all_programs_on_single_demonstration.
    """
    X, y = None, None

    for demo_number in demo_numbers:
        demo_X, demo_y = run_all_programs_on_single_demonstration(base_class_name, num_programs, demo_number, interactive=interactive)

        if X is None:
            X = demo_X
            y = demo_y
        else:
            X = vstack([X, demo_X])
            y.extend(demo_y)

    y = np.array(y, dtype=np.uint8)

    return X, y

def learn_single_batch_decision_trees(y, num_dts, X_i):
    """
    Parameters
    ----------
    y : [ bool ]
    num_dts : int
    X_i : csr_matrix

    Returns
    -------
    clfs : [ DecisionTreeClassifier ]
    """
    clfs = []

    for seed in range(num_dts):
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_i, y)
        clfs.append(clf)

    return clfs

def learn_plps(X, y, programs, program_prior_log_probs, num_dts=5, program_generation_step_size=10):
    """
    Parameters
    ----------
    X : csr_matrix
    y : [ bool ]
    programs : [ StateActionProgram ]
    program_prior_log_probs : [ float ]
    num_dts : int
    program_generation_step_size : int

    Returns
    -------
    plps : [ StateActionProgram ]
    plp_priors : [ float ]
        Log probabilities.
    """
    plps = []
    plp_priors = []

    num_programs = len(programs)

    for i in range(0, num_programs, program_generation_step_size):
        print("Learning plps with {} programs".format(i))
        for clf in learn_single_batch_decision_trees(y, num_dts, X[:, :i+1]):
            plp, plp_prior_log_prob = extract_plp_from_dt(clf, programs, program_prior_log_probs)
            plps.append(plp)
            plp_priors.append(plp_prior_log_prob)
        
    print("Learn all probabilities!")
    return plps, plp_priors

def compute_likelihood_single_plp(demonstrations, plp):
    """
    Parameters
    ----------
    demonstrations : [(np.ndarray, (int, int))]
        State, action pairs.
    plp : StateActionProgram
    
    Returns
    -------
    likelihood : float
        The log likelihood.
    """
    ll = 0.
    # state, loc_and_action = demonstration_item
    # a, loc = action_and_loc

    # positive_examples = [(state, loc, a)]
    # negative_examples = []

    for obs, action_and_loc in demonstrations:
        a, loc = action_and_loc
        if not plp(obs, loc, a):
            return -np.inf
        
        # if a != "xyz.PASS":
        #     import pickle
        #     f = open('test.pkl', 'wb')
        #     pickle.dump([plp, obs, action_and_loc], f)
        #     f.close()
        size = 1
        for r in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                for act in ('pass', 'x', 'y','z'):
                    if (r,c) == loc and act==a:
                        continue
                    if plp(obs, (r, c), a):
                        size += 1
                    

        ll += np.log(1. / size)

    return ll

def compute_likelihood_plps(plps, demonstrations):
    """
    See compute_likelihood_single_plp.
    """
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)

    fn = partial(compute_likelihood_single_plp, demonstrations)
    likelihoods = pool.map(fn, plps)
    pool.close()

    return likelihoods

def select_particles(particles, particle_log_probs, max_num_particles):
    """
    Parameters
    ----------
    particles : [ Any ]
    particle_log_probs : [ float ]
    max_num_particles : int

    Returns
    -------
    selected_particles : [ Any ]
    selected_particle_log_probs : [ float ]
    """
    sorted_log_probs, _, sorted_particles = (list(t) \
        for t in zip(*sorted(zip(particle_log_probs, np.random.random(size=len(particles)), particles), reverse=True)))
    end = min(max_num_particles, len(sorted_particles))
    try:
        idx = sorted_log_probs.index(-np.inf)
        end = min(idx, end)
    except ValueError:
        pass
    return sorted_particles[:end], sorted_log_probs[:end]

@manage_cache(cache_dir, '.pkl')
def train(base_class_name, demo_numbers, program_generation_step_size, num_programs, num_dts, max_num_particles, interactive=False):
    programs, program_prior_log_probs = get_program_set(base_class_name, num_programs)

    X, y = run_all_programs_on_demonstrations(base_class_name, num_programs, demo_numbers, interactive)
    plps, plp_priors = learn_plps(X, y, programs, program_prior_log_probs, num_dts=num_dts,
        program_generation_step_size=program_generation_step_size)
    if base_class_name == "PlayingWithXYZ": demonstrations = get_demonstrations(base_class_name, demo_numbers=demo_numbers, max_demo_length=2,interactive=interactive)
    else: demonstrations = get_demonstrations(base_class_name, demo_numbers=demo_numbers)
    print("Starting to compute the likelihood")
    likelihoods = compute_likelihood_plps(plps, demonstrations)
    print("Likelihood calculation completed")
    # import pickle
    # f = open('plps.pkl', 'wb')
    # pickle.dump([plps,plp_priors], f)
    # f.close()

    particles = []
    particle_log_probs = []

    for plp, prior, likelihood in zip(plps, plp_priors, likelihoods):
        particles.append(plp)
        particle_log_probs.append(prior + likelihood)


    print("\nDone!")
    map_idx = np.argmax(particle_log_probs).squeeze()
    print("MAP program ({}):".format(particle_log_probs[map_idx]))
    print(particles[map_idx])

    top_particles, top_particle_log_probs = select_particles(particles, particle_log_probs, max_num_particles)
    if len(top_particle_log_probs) > 0:
        top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(top_particle_log_probs)
        top_particle_probs = np.exp(top_particle_log_probs)
        print("top_particle_probs:", top_particle_probs)
        policy = PLPPolicy(top_particles, top_particle_probs)
    else:
        print("no nontrivial particles found")
        policy = PLPPolicy([StateActionProgram("False")], [1.0])

    return policy

## Test (given subset of environments)
def test(policy, base_class_name, test_env_nums=range(4), max_num_steps=3,
         record_videos=True, video_format='mp4'):
    
    env_names = ['{}{}-v0'.format(base_class_name, i) for i in test_env_nums]
    envs = [gym.make(env_name) for env_name in env_names]
    accuracies = []
    
    for env in envs:
        video_out_path = '/tmp/lfd_{}.{}'.format(env.__class__.__name__, video_format)
        result = run_single_episode(env, policy, max_num_steps=max_num_steps, 
            record_video=record_videos, video_out_path=video_out_path)
        accuracies.append(result['accuracies'])


    return accuracies

if __name__  == "__main__":
    policy = train("PlayingWithXYZ", range(0,3), 1, 500, 5, 25, interactive=True )
    test_results = test(policy, "PlayingWithXYZ", range(0,4), record_videos=False)
    print("Test results:", test_results)
