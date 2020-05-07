from cache_utils import manage_cache
from dsl import *
from env_settings import *
from grammar_utils import generate_programs#, generate_programs_test
from dt_utils import extract_plp_from_dt
from expert_demonstrations import get_demonstrations, get_interactive_demo, unity_demontration
from policy import StateActionProgram, PLPPolicy
from utils import run_single_episode
from generalization_grid_games.envs import playing_with_XYZ
from itertools import product
import random
from collections import defaultdict
from functools import partial
from sklearn.tree import DecisionTreeClassifier
from scipy.special import logsumexp
#from scipy.stats import bernoulli
from scipy.sparse import csr_matrix, lil_matrix, vstack
from copy import deepcopy
import gym
import multiprocessing
import numpy as np
import time
import os
from sklearn.model_selection import cross_validate
import matplotlib
import UnityDemo.UnityVisualization
from numpy.random import default_rng

#matplotlib.use('TkAgg')

def pipeline_manager(cache_dir,cache_program,cache_matrix,useCache):
    # cache_dir = 'cache'
    # #useCache = False
    # cache_program = False
    # cache_matrix = False and cache_program
    # useCache = False and cache_matrix

    @manage_cache(cache_dir, ['.pkl', '.pkl'], enabled = cache_program)
    def get_program_set(base_class_name, num_programs, test_dimension = None):
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
        if test_dimension == "reduced" and base_class_name=="UnityGame":
            object_types = get_object_types("UnityGame_reduced")
        if base_class_name == "UnityGame":
            object_types = object_types #
            grammar = create_grammar_unity(object_types) 
        else:
            grammar = create_grammar(object_types)

        # if base_class_name=="PlayingWithXYZ":
        #     program_generator = generate_programs_test(grammar)

        program_generator = generate_programs(grammar,game_class=base_class_name)
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
        object_types = playing_with_XYZ.ALL_ACTION_TOKENS   
        
        @classmethod
        def extract_examples_from_demonstration(cls,demonstration):
            positive_examples = []
            negative_examples = []
            start = time.time()
            for idx, demonstration_item in enumerate(demonstration):
                #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_TEST(idx,demonstration)
                #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item(demonstration_item)
                demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_sample(demonstration_item)
                positive_examples.extend(demo_positive_examples)
                negative_examples.extend(demo_negative_examples)
            print("Total time for generating demonstrations: {}".format(time.time()-start))
            return positive_examples, negative_examples
        
        @classmethod
        def extract_examples_from_demonstration_item_sample(cls,demonstration_item):
            state, loc_and_action = demonstration_item
            a, loc = loc_and_action

            positive_examples = [(state, loc, a)]
            negative_examples = []
            try:
                #Sample 250 hundred random demonstrations instead of computing all of them
                indeces = random.sample(list(product(range(state.shape[0]),range(state.shape[1]),['x','y','z','empty','pass'])),250)

                for nums in indeces:
                    if nums[:2] == loc and nums[2] == a: 
                        continue
                    else:
                        negative_examples.append((state, nums[:2], nums[2]))
                return positive_examples, negative_examples
            except:
                print("Skipped as no so many demonstrations")
                for r in range(state.shape[0]):
                    for c in range(state.shape[1]):
                        for val in cls.object_types:
                            if val == a and (r, c) == loc:
                                continue
                            else:
                                negative_examples.append((state, (r, c), val))

                return positive_examples, negative_examples

    # class UnityGame:
    #     object_types = playing_with_XYZ.ALL_ACTION_TOKENS   
        
    #     @classmethod
    #     def extract_examples_from_demonstration(cls,demonstration):
    #         positive_examples = []
    #         negative_examples = []
    #         start = time.time()
    #         for idx, demonstration_item in enumerate(demonstration):
    #             #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_TEST(idx,demonstration)
    #             #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item(demonstration_item)
    #             demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_sample(demonstration_item)
    #             positive_examples.extend(demo_positive_examples)
    #             negative_examples.extend(demo_negative_examples)
    #         print("Total time for generating demonstrations: {}".format(time.time()-start))
    #         return positive_examples, negative_examples
        
    #     @classmethod
    #     def extract_examples_from_demonstration_item_sample(cls,demonstration_item):
    #         state, loc_and_action = demonstration_item
    #         a, loc = loc_and_action

    #         positive_examples = [(state, loc, a)]
    #         negative_examples = []
    #         try:
    #             #Sample 250 hundred random demonstrations instead of computing all of them
    #             indeces = random.sample(list(product(range(state.shape[0]),range(state.shape[1]),['x','y','z','empty','pass'])),250)

    #             for nums in indeces:
    #                 if nums[:2] == loc and nums[2] == a: 
    #                     continue
    #                 else:
    #                     negative_examples.append((state, nums[:2], nums[2]))
    #             return positive_examples, negative_examples
    #         except:
    #             print("Skipped as no so many demonstrations")
    #             for r in range(state.shape[0]):
    #                 for c in range(state.shape[1]):
    #                     for val in cls.object_types:
    #                         if val == a and (r, c) == loc:
    #                             continue
    #                         else:
    #                             negative_examples.append((state, (r, c), val))

    #             return positive_examples, negative_examples

        # @classmethod
        # def extract_examples_from_demonstration_item_TEST(cls,idx,demonstration):
        #     state, loc_and_action = demonstration[idx]
        #     a, loc = loc_and_action

        #     positive_examples = [(state, loc, a)]
        #     negative_examples = []

        #     for r in range(state.shape[0]):
        #         for c in range(state.shape[1]):
        #             for val in cls.object_types:
        #                 # if (val, (r,c)) in list(zip(*demonstration))[1]:
        #                 #     continue
        #                 if val == a and (r,c) in [x[1] for x in list(zip(*demonstration[idx:]))[1] if x[0] == a]: 
        #                     #Assume serializable actions
        #                     positive_examples.append((state, (r, c), val))
        #                     continue
        #                 else:
        #                     negative_examples.append((state, (r, c), val))

        #     return positive_examples, negative_examples
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

        #         acton_inputs[idx] = list(acton_inputs[iDemonstrationHandler   acton_inputs[idx][1] = acton_inputs[idx][1][0]
        #         acton_inputs[idx] = tuple(acton_inputs[idx])
        #     return fn_inputs, acton_inputs

    # class UnityGame:
    #     object_types = playing_with_XYZ.ALL_ACTION_TOKENS   
        

    #     @classmethod
    #     def extract_examples_from_demonstration(cls,demonstration):
    #         positive_examples = []
    #         negative_examples = []
    #         start = time.time()
    #         for idx, demonstration_item in enumerate(demonstration):
    #             #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_TEST(idx,demonstration)
    #             #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item(demonstration_item)
    #             demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_sample(demonstration_item)
    #             positive_examples.extend(demo_positive_examples)
    #             negative_examples.extend(demo_negative_examples)
    #         print("Total time for generating demonstrations: {}".format(time.time()-start))
    #         return positive_examples, negative_examples
        
    #     @classmethod
    #     def extract_examples_from_demonstration_item_sample(cls,demonstration_item):
    #         state, loc_and_action = demonstration_item
    #         a, loc = loc_and_action

    #         positive_examples = [(state, loc, a)]
    #         negative_examples = []
    #         try:
    #             #Sample 250 hundred random demonstrations instead of computing all of them
    #             indeces = random.sample(list(product(range(state.shape[0]),range(state.shape[1]),['x','y','z','empty','pass'])),250)

    #             for nums in indeces:
    #                 if nums[:2] == loc and nums[2] == a: 
    #                     continue
    #                 else:
    #                     negative_examples.append((state, nums[:2], nums[2]))
    #             return positive_examples, negative_examples
    #         except:
    #             print("Skipped as no so many demonstrations")
    #             for r in range(state.shape[0]):
    #                 for c in range(state.shape[1]):
    #                     for val in cls.object_types:
    #                         if val == a and (r, c) == loc:
    #                             continue
    #                         else:
    #                             negative_examples.append((state, (r, c), val))

    #             return positive_examples, negative_examples




    @manage_cache(cache_dir, ['.npz', '.pkl'], enabled = cache_matrix)
    def run_all_programs_on_single_demonstration(base_class_name, num_programs, demo_number, program_interval=1000, interactive=False, specify_task = None,test_dimension=None):
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

        programs, _ = get_program_set(base_class_name, num_programs, test_dimension= test_dimension)
        

        #Max demo needed because a demostration is considered as well doing nothing. Hence legth is set
        if base_class_name == "PlayingWithXYZ":
            demonstration = get_demonstrations(base_class_name, demo_numbers=(demo_number,),  max_demo_length=1, interactive=interactive)
            positive_examples, negative_examples  = PlayingWithXYZ.extract_examples_from_demonstration(demonstration)
        elif base_class_name == "UnityGame":
            demonstration = unity_demontration(demo_number, specify_task=specify_task)
            positive_examples, negative_examples  = demonstration.extract_examples_from_demonstration()
            #Visualize demostration 
            #UnityDemo.UnityVisualization.UnityVisualization(negative_examples).visualize_demonstration()
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

        start = time.time()

        #Debugging
        #program_interval = 200

        # This loop avoids memory issues
        for i in range(l, num_programs, program_interval):
            end = min(i+program_interval, num_programs)
            print('Iteration {} of {}'.format(i, num_programs), end='\r')

            
            
            #Debugging
            # fn = partial(apply_programs, programs[i:end])
            # if i == 49:
            #     print("Hello2")
            # results = list(map(fn, fn_inputs))
            # print("Miao")
            #Multiprocessing
            num_workers = multiprocessing.cpu_count()        
            pool = multiprocessing.Pool(num_workers)
            fn = partial(apply_programs, programs[i:end])
            results = pool.map(fn, fn_inputs)
            pool.close()
            #print("Hello")
            for X_idx, x in enumerate(results):
                X[X_idx, i:end] = x
        print("Total time for generating programs: {}".format(time.time()-start))
        X = X.tocsr()

        print()
        return X, y

    def run_all_programs_on_demonstrations(base_class_name, num_programs, demo_numbers, interactive=False, specify_task = None, test_dimension=None):
        """
        See run_all_programs_on_single_demonstration.
        """
        X, y = None, None

        for demo_number in demo_numbers:
            demo_X, demo_y = run_all_programs_on_single_demonstration(base_class_name, num_programs, demo_number, interactive=interactive, specify_task=specify_task, test_dimension=test_dimension)

            if X is None:
                X = demo_X
                y = demo_y
            else:
                X = vstack([X, demo_X])
                y.extend(demo_y)

        y = np.array(y, dtype=np.uint8)

        return X, y

    def new_learn_single_batch_decision_trees(y, num_dts, X_i):
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
        numbers = [1]
        for seed in range(num_dts):
            #X_i_curr = X_i
            #rng = default_rng()
            #numbers = rng.choice(X_i_curr.shape[1], size=int(X_i_curr.shape[1]/(num_dts*2)*seed), replace=False)
            #rng = default_rng()
            #numbers = rng.choice(X_i_curr.shape[1], size=int(X_i_curr.shape[1]/(num_dts*5)*seed), replace=False)
            # if len(numbers!=0): 
            #     #niente = np.delete(X_i_curr.toarray(), numbers, axis=1) 
            #      X_i_curr = lil_matrix(X_i_curr)
            #      X_i_curr[:,numbers] = False
            clf = DecisionTreeClassifier(splitter="random", max_features="log2", random_state=seed*6)
            cv_results = cross_validate(clf,X_i,y, return_estimator=True,cv=3, return_train_score=True)
            if cv_results['test_score'].max() == 1 and cv_results['train_score'][cv_results['test_score'].argmax()] == 1:
                 res = cv_results['estimator'][cv_results['test_score'].argmax()]
                 clfs.append([res,cv_results])

            # clf.fit(X_i_curr, y)
            # if clf.score(X_i_curr, y) == 1:
            #     clfs.append([clf,0])
            #res = cv_results['estimator'][cv_results['test_score'].argmax()]
            #clfs.append([res,cv_results])
        return clfs


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
        numbers = [1]
        for seed in range(num_dts):
            X_i_curr = X_i
            rng = default_rng()
            numbers = rng.choice(X_i_curr.shape[1], size=int(X_i_curr.shape[1]/(num_dts*5)*seed), replace=False)
            if len(numbers!=0): 
                #niente = np.delete(X_i_curr.toarray(), numbers, axis=1) 
                X_i_curr = lil_matrix(X_i_curr)
                X_i_curr[:,numbers] = False
            clf = DecisionTreeClassifier()
            # cv_results = cross_validate(clf,X_i_curr,y, return_estimator=True,cv=10, return_train_score=True)
            # if cv_results['test_score'].max() == 1 and cv_results['train_score'][cv_results['test_score'].argmax()] == 1:
            #     res = cv_results['estimator'][cv_results['test_score'].argmax()]
            #     clfs.append([res,cv_results])
            clf.fit(X_i_curr, y)
            if clf.score(X_i_curr, y) == 1:
                clfs.append([clf,0])
            #res = cv_results['estimator'][cv_results['test_score'].argmax()]
            #clfs.append([res,cv_results])
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
        likelihood = []
        total_leaves = []
        variances = []
        clf_tot = []
        #num_features = []

        num_programs = len(programs)

        # for i in range(0, num_programs, program_generation_step_size):
        #     print("Learning plps with {} programs".format(i))

        for clf in new_learn_single_batch_decision_trees(y, num_dts, X):
            plp, plp_prior_log_prob, likelihood_prob, total_leaf, variance = extract_plp_from_dt(clf, programs, program_prior_log_probs, len([seq for seq in y if seq==1]))
            likelihood.append(likelihood_prob)
            plps.append(plp)
            plp_priors.append(plp_prior_log_prob)
            total_leaves.append(total_leaf)
            variances.append(variance)
            clf_tot.append(clf[0])
            #num_features.append(i)

        
        print("Leart all probabilities!")
        return plps, plp_priors, likelihood, total_leaves, variances, clf_tot

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
        #demon = 1
        #start = time.time()
        for obs, action_and_loc in demonstrations:
            a, loc = action_and_loc
            if not plp(obs, loc, a):
                return -np.inf
            #when_false = when_false + 1

        
        for obs, action_and_loc in demonstrations:
            #print("Starting analyzing demonstration {}".format(demon))
            #start = time.time()
            #demon = demon + 1 
            a, loc = action_and_loc
            # if not plp(obs, loc, a):
            #     print(when_false)
            #     return -np.inf
            # if a != "xyz.PASS":
            #     import pickle
            #     f = open('test.pkl', 'wb')
            #     pickle.dump([plp, obs, action_and_loc], f)
            #     f.close()
            size = 1
            for r in range(obs.shape[0]):
                for c in range(obs.shape[1]):
                    for act in playing_with_XYZ.ALL_ACTION_TOKENS:
                        if (r,c) == loc and act==a:
                            continue
                        if plp(obs, (r, c), act):
                            size += 1
                        
            #print("Demonstration {} done in {}".format(demon, time.time() - start))
            ll += np.log(1. / size)
            #when_false = when_false + 1

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

    def filter_negative_demonstrations(design_matrix,output):
        """
        Another sampling from negative distribution to filter out false positives
        """
        isSkip = np.zeros(len(design_matrix.toarray()))
        design_matrix = design_matrix.toarray()
        positive_design_matrix = design_matrix[output==1]
        negative_design_matrix = design_matrix[output==0]
        positive_mean = np.var(positive_design_matrix,axis=0)
        positive_var = np.mean(positive_design_matrix,axis=0)
        negative_mean = np.var(negative_design_matrix,axis=0)
        negative_var = np.mean(negative_design_matrix,axis=0)
        #Kick out the 50 demonstrations with lowest probability

        for x in range(len(isSkip)):
            if design_matrix[x,-1] == 1:
                continue
            isSkip[x] = check_negative_example(design_matrix[x],positive_mean,threeshold=0.2,n_max=5)

        design_matrix = design_matrix[isSkip == 0]
        output = output[isSkip==0]
        #design_matrix = design_matrix[1]
        #design_matrix = design_matrix[2]
        return design_matrix,output



    def check_negative_example(X_row,positive_mean,threeshold=0.2,n_max=5):
        """
        Use Bernoulli to check out which probability to keep
        """
        skip = 0
        total = np.sum(X_row[(X_row*positive_mean) > threeshold])
        if total > n_max:
            skip=1
            return skip
        else:
            return skip

    def get_rid_of_pointless_rows(X,programs,program_prior_log_probs):
        X = X.toarray()
        to_delete = np.all(X  == X[0,:], axis = 0)
        to_keep = np.logical_not(to_delete)
        X = X[:,to_keep].squeeze()
        programs = [programs[idx] for idx, x in enumerate(to_keep) if x == True ]
        program_prior_log_probs = [program_prior_log_probs[idx] for idx, x in enumerate(to_keep) if x == True ]
        return X, programs, program_prior_log_probs

        


    @manage_cache(cache_dir, '.pkl', enabled = useCache)
    def train(base_class_name, demo_numbers, program_generation_step_size, num_programs, num_dts = None ,max_num_particles = None, interactive=False, specify_task = None, test_dimension=None):
        programs, program_prior_log_probs = get_program_set(base_class_name, num_programs, test_dimension=test_dimension)
        import json
        with open('debug/programs.json', 'w', encoding='utf-8') as f:
            string_programs = [current_progr.program for current_progr in programs]
            json.dump(string_programs, f, ensure_ascii=False, indent=4)
        X, y = run_all_programs_on_demonstrations(base_class_name, num_programs, demo_numbers, interactive, specify_task, test_dimension=test_dimension)
        # with open('debug/X.json', 'w', encoding='utf-8') as f:
        #     string_programs = X.toarray()
        #     json.dump(string_programs, f, ensure_ascii=False, indent=4)
        # with open('debug/y.json', 'w', encoding='utf-8') as f:
        #     string_programs = y
        #     json.dump(string_programs, f, ensure_ascii=False, indent=4)
        #X, y = filter_negative_demonstrations(X,y)
        X,programs,program_prior_log_probs = get_rid_of_pointless_rows(X,programs,program_prior_log_probs)

        plps, plp_priors,likelihood, total_leaves, variance, clf_tot = learn_plps(X, y, programs, program_prior_log_probs, num_dts=num_dts,
            program_generation_step_size=program_generation_step_size)
        #if base_class_name == "PlayingWithXYZ": demonstrations = get_demonstrations(base_class_name, demo_numbers=demo_numbers, max_demo_length=2,interactive=interactive)
        #else: demonstrations = get_demonstrations(base_class_name, demo_numbers=demo_numbers)
        #print("Starting to compute the likelihood")
        #likelihoods = compute_likelihood_plps(plps, demonstrations)
        #print("Likelihood calculation completed")
        #print("Results of likelihood: {}".format(likelihood))
        # import pickle
        # f = open('plps.pkl', 'wb')
        # pickle.dump([plps,plp_priors], f)
        # f.close()

        particles = []
        particle_log_probs = []

        for plp, prior_scalar, likelihood_scalar, curr_variance in zip(plps, plp_priors, likelihood, variance):
            #print("Prior: {}".format(prior))
            particles.append(plp)
            #print("Likelihood: {}".format(likelihood))
            particle_log_probs.append(-prior_scalar + likelihood_scalar)
            #particle_log_probs.append(prior_scalar + curr_variance )
            #print("Posterior: {}".format(prior + 500*likelihood))


        print("\nDone!")
        if len(particle_log_probs) != 0:
            map_idx = np.argmax(particle_log_probs).squeeze()
        else: 
            print("No solution found!")
            policy = PLPPolicy([StateActionProgram("False")], [1.0],base_class=base_class_name)
            return policy
        #map_idx = np.argmax(variance).squeeze()
        print("MAP program ({}):".format(particle_log_probs[map_idx]))
        print("Likelihood {}".format(likelihood[map_idx]))
        print("Prior {}".format(plp_priors[map_idx]))
        print("Total leaves {}".format(total_leaves[map_idx]))
        print("Variance: {}".format(variance[map_idx]))
        print(particles[map_idx])
        print("Tree Learnt {}".format(clf_tot[map_idx]))
        #TEST Analyzing wrong demonstrations
        #extract_plp_from_dt(clf_tot[map_idx], programs, program_prior_log_probs, len([seq for seq in y if seq==1]))

        top_particles, top_particle_log_probs = select_particles(particles, particle_log_probs, max_num_particles)
        if len(top_particle_log_probs) > 0:
            top_particle_log_probs = np.array(top_particle_log_probs) - logsumexp(top_particle_log_probs)
            top_particle_probs = np.exp(top_particle_log_probs)
            print("top_particle_probs:", top_particle_probs)
            policy = PLPPolicy(top_particles, top_particle_probs,base_class=base_class_name)
        else:
            print("no nontrivial particles found")
            policy = PLPPolicy([StateActionProgram("False")], [1.0],base_class=base_class_name)

        return policy
    return train
## Test (given subset of environments)
def test(policy, base_class_name, test_env_nums=range(4), max_num_steps=20,
        record_videos=True, video_format='mp4', interactive = True):
    
    env_names = ['{}{}-v0'.format(base_class_name, i) for i in test_env_nums]
    envs = [gym.make(env_name) for env_name in env_names]
    accuracies = []
    
    for env in envs:
        video_out_path = 'video/lfd_{}.{}'.format(env.__class__.__name__, video_format)
        result = run_single_episode(env, policy, max_num_steps=max_num_steps, 
            record_video=record_videos, video_out_path=video_out_path, base_class= base_class_name)

        accuracies.append(result['accuracy'])


    return accuracies

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

    # def interactive_learning(policy, base_class_name, test_env_nums=range(4), max_num_steps=3):
        
    #     env_names = ['{}{}-v0'.format(base_class_name, i) for i in test_env_nums]
    #     envs = [gym.make(env_name) for env_name in env_names]
    #     accuracies = []
        
    #     for env in envs:
    #         result = run_single_episode(env, policy, max_num_steps=max_num_steps, 
    #             record_video=record_videos, video_out_path=video_out_path)
    #         if result['accuracy'] == None and res['Unkown Observation']:
    #             train("PlayingWithXYZ", range(0,3) + env, 1, 500, 5, 25, interactive=True )
    #         accuracies.append(result['accuracy'])


if __name__  == "__main__":

    cache_dir = 'cache'
    cache_program = True
    cache_matrix = True and cache_program
    useCache = False and cache_matrix
    #train("TwoPileNim", range(11), 1, 31, 100, 25)
    #policy = train("UnityGame", range(0,4), 50, 1000, num_dts= 500, max_num_particles = 5, interactive=True, specify_task="Put_obj_in_boxes" )
    train = pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
    policy = train("UnityGame", range(0,3), 200, 300, 300, 5, interactive=True, specify_task="Naive_game",test_dimension="reduced" )
    #policy = interactive_learning()
    test_results = test(policy, "UnityGame", range(1,2), record_videos=True, interactive = False)
    #print("Test results:", test_results)
