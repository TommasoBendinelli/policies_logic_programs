import unittest 
import pipeline
import gym
import numpy as np
import sys
import os
from ray.tune import CLIReporter

import numpy as np
import ray
from ray import tune
from ray.tune import track
from ray.tune import Trainable, run, sample_from

class MyTrainableClass(Trainable):

    def _train(self):
        cache_dir = 'cache'
        useCache = False
        cache_program = False
        cache_matrix = False and cache_program
        useCache = False and cache_matrix
        train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = train("UnityGame", range(0,3), 20, 300, 200, 5, interactive=True, specify_task="Naive_game", test_dimension="reduced" )
        sys.stdout = save_stdout
        res_experiment = dict()
        res_experiment = test_layer_1(self,policy,res_experiment)
        res_experiment = test_layer_2(self,policy,res_experiment)
        return res_experiment

    def test_layer_1(self,policy,res_experiment):
        results1 = ["S","B",None]
        env_names = 'UnityGame0-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(3):
            action = policy(obs)
            if obs[action] != results1[t]: 
                res_experimemt["result1"] = -1
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs
        if not res_experiment["result1"]:
            res_experimemt["result1"] = 1
        env.close()

    def test_layer_2(self,policy,res_experiment):
        env_names = 'UnityGame1-v0'
        env = gym.make(env_names)
        obs = env.reset()
        results2 = ["S","B","S","B","S","B","S","B","S","B",None]   
        for t in range(11):
            action = policy(obs)
            if obs[action] != results2[t]:
                res_experimemt["result2"] = -1
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs
        if not res_experiment["result2"]:
            res_experimemt["result2"] = 1

        env.close()
        return res_experiment

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--smoke-test", action="store_true", help="Finish quickly for testing")
    # parser.add_argument(
    #     "--ray-address",
    #     help="Address of Ray cluster for seamless distributed execution.")
    # args, _ = parser.parse_known_args()
    # ray.init(address=args.ray_address)

    # asynchronous hyperband early stopping, configured with
    # `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    # ahb = AsyncHyperBandScheduler(
    #     time_attr="training_iteration",
    #     metric="episode_reward_mean",
    #     mode="max",
    #     grace_period=3,
    #     max_t=100)

    reporter = CLIReporter()
    reporter.add_metric_column("result1")
    reporter.add_metric_column("result2")


    #logger = logger.Logger()

    search_space = {
        "number_of_tree":,
        "dropout_difference":,
        "CV_or_tree":,
        "droput_ratio":,
        }

    run(MyTrainableClass,
        name="Test Siamese",
        progress_reporter=reporter,
        num_samples=1,
        loggers=[TBXLogger1],
        stop={"training_iteration": 1},
        config=search_space,
        local_dir="experiments/")

# class TestFirstGame(sunittest.TestCase):

#     def test_layer1(self):
#         results = ["S","B",None]
#         cache_dir = 'cache'
#         useCache = False
#         cache_program = False
#         cache_matrix = False and cache_program
#         useCache = False and cache_matrix
#         train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
#         save_stdout = sys.stdout
#         sys.stdout = open('trash', 'w')
#         policy = train("UnityGame", range(0,3), 20, 300, 200, 5, interactive=True, specify_task="Naive_game", test_dimension="reduced" )
#         sys.stdout = save_stdout
#         env_names = 'UnityGame0-v0'
#         env = gym.make(env_names)
#         obs = env.reset()
#         total_reward = 0.
#         for t in range(3):
#             action = policy(obs)
#             self.assertEqual(obs[action],results[t])
#             new_obs, reward, done, debug_info = env.step(action)
#             obs = new_obs

#         env.close()

#     def test_layer2(self):
#         results = ["S","B","S","B","S","B","S","B","S","B",None]
#         cache_dir = 'cache'
#         useCache = False
#         cache_program = False
#         cache_matrix = False and cache_program
#         useCache = False and cache_matrix
#         train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
#         save_stdout = sys.stdout
#         sys.stdout = open('trash', 'w')
#         policy = train("UnityGame", range(0,3), 20, 300, 200, 5, interactive=True, specify_task="Naive_game",test_dimension="reduced"  )
#         sys.stdout = save_stdout
#         env_names = 'UnityGame1-v0'
#         env = gym.make(env_names)
#         obs = env.reset()
#         total_reward = 0.
#         for t in range(11):
#             action = policy(obs)
#             self.assertEqual(obs[action],results[t])
#             new_obs, reward, done, debug_info = env.step(action)
#             obs = new_obs

#         env.close()

#     #Test with big number of objs
#     def test_layer1_big(self):
#         results = ["S","B",None]
#         cache_dir = 'cache'
#         useCache = False
#         cache_program = False
#         cache_matrix = False and cache_program
#         useCache = False and cache_matrix
#         train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
#         save_stdout = sys.stdout
#         sys.stdout = open('trash', 'w')
#         policy = train("UnityGame", range(0,3), 200, 4000, 200, 5, interactive=True, specify_task="Naive_game" )
#         sys.stdout = save_stdout
#         env_names = 'UnityGame0-v0'
#         env = gym.make(env_names)
#         obs = env.reset()
#         total_reward = 0.
#         for t in range(3):
#             action = policy(obs)
#             print(action)
#             self.assertEqual(obs[action],results[t])
#             new_obs, reward, done, debug_info = env.step(action)
#             obs = new_obs

#         env.close()

#     #Test with big number of objs
#     def test_layer2_big(self):
#         results = ["S","B","S","B","S","B","S","B","S","B",None]
#         cache_dir = 'cache'
#         useCache = False
#         cache_program = False
#         cache_matrix = False and cache_program
#         useCache = False and cache_matrix
#         train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
#         save_stdout = sys.stdout
#         sys.stdout = open('trash', 'w')
#         policy = train("UnityGame", range(0,3), 200, 4000, 200, 5, interactive=True, specify_task="Naive_game" )
#         sys.stdout = save_stdout
#         env_names = 'UnityGame1-v0'
#         env = gym.make(env_names)
#         obs = env.reset()
#         total_reward = 0.
#         for t in range(11):
#             action = policy(obs)
#             print(action)
#             self.assertEqual(obs[action],results[t])
#             new_obs, reward, done, debug_info = env.step(action)
#             obs = new_obs

#         env.close()
    
# class TestSecondGame(unittest.TestCase):

#     @staticmethod
#     def check_equality(obj):
#         if obj in ["S","CBLA","CB"]:
#             return True
#         else:
#             return False

#     def test_put_black_blue_siemens_in_box(self):
#         results = ["S","B",None]
#         cache_dir = 'cache'
#         useCache = False
#         cache_program = False
#         cache_matrix = False and cache_program
#         useCache = False and cache_matrix
#         train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
#         save_stdout = sys.stdout
#         sys.stdout = open('trash', 'w')
#         policy = train("UnityGame", range(0,4), 200, 4000, 200, 5, interactive=True, specify_task="Put_obj_in_boxes")
#         sys.stdout = save_stdout
#         env_names = 'UnityGame2-v0'
#         env = gym.make(env_names)
#         obs = env.reset()
#         total_reward = 0.
#         for t in range(17):
#             action = policy(obs)
#             print(action)
#             if t%2 == 1:
#                 self.assertEqual(obs[action],"B")
#             if t%2 == 0 & t != 16:
#                 res = self.check_equality(action)
#                 self.assertEqual(True,True)
#             if t == 16:
#                 self.assertEqual(obs[action],None)
#             new_obs, reward, done, debug_info = env.step(action)
#             obs = new_obs

#         env.close()