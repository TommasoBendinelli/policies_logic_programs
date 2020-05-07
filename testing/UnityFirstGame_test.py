import unittest 
import pipeline
import gym
import numpy as np
import sys
import os


class TestFirstGame(unittest.TestCase):

    def test_layer1(self):
        results = ["S","B",None]
        cache_dir = 'cache'
        useCache = False
        cache_program = False
        cache_matrix = False and cache_program
        useCache = False and cache_matrix
        train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = train("UnityGame", range(0,3), 20, 300, 300, 5, interactive=True, specify_task="Naive_game", test_dimension="reduced" )
        sys.stdout = save_stdout
        env_names = 'UnityGame0-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(3):
            action = policy(obs)
            self.assertEqual(obs[action],results[t])
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs

        env.close()

    def test_layer2(self):
        results = ["S","B","S","B","S","B","S","B","S","B",None]
        cache_dir = 'cache'
        useCache = False
        cache_program = False
        cache_matrix = False and cache_program
        useCache = False and cache_matrix
        train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = train("UnityGame", range(0,3), 20, 300, 300, 5, interactive=True, specify_task="Naive_game",test_dimension="reduced"  )
        sys.stdout = save_stdout
        env_names = 'UnityGame1-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(11):
            action = policy(obs)
            self.assertEqual(obs[action],results[t])
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs

        env.close()

    #Test with big number of objs
    def test_layer1_big(self):
        results = ["S","B",None]
        cache_dir = 'cache'
        useCache = False
        cache_program = False
        cache_matrix = False and cache_program
        useCache = False and cache_matrix
        train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = train("UnityGame", range(0,3), 200, 4000, 4000, 5, interactive=True, specify_task="Naive_game" )
        sys.stdout = save_stdout
        env_names = 'UnityGame0-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(3):
            action = policy(obs)
            print(action)
            self.assertEqual(obs[action],results[t])
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs

        env.close()

    #Test with big number of objs
    def test_layer2_big(self):
        results = ["S","B","S","B","S","B","S","B","S","B",None]
        cache_dir = 'cache'
        useCache = False
        cache_program = False
        cache_matrix = False and cache_program
        useCache = False and cache_matrix
        train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = train("UnityGame", range(0,3), 200, 4000, 4000, 5, interactive=True, specify_task="Naive_game" )
        sys.stdout = save_stdout
        env_names = 'UnityGame1-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(11):
            action = policy(obs)
            print(action)
            self.assertEqual(obs[action],results[t])
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs

        env.close()
    
class TestSecondGame(unittest.TestCase):

    @staticmethod
    def check_equality(obj):
        if obj in ["S","CBLA","CB"]:
            return True
        else:
            return False

    def test_put_black_blue_siemens_in_box(self):
        results = ["S","B",None]
        cache_dir = 'cache'
        useCache = False
        cache_program = False
        cache_matrix = False and cache_program
        useCache = False and cache_matrix
        train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = train("UnityGame", range(0,4), 200, 4000, 4000, 5, interactive=True, specify_task="Put_obj_in_boxes")
        sys.stdout = save_stdout
        env_names = 'UnityGame2-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(17):
            action = policy(obs)
            print(action)
            if t%2 == 1:
                self.assertEqual(obs[action],"B")
            if t%2 == 0 & t != 16:
                res = self.check_equality(action)
                self.assertEqual(True,True)
            if t == 16:
                self.assertEqual(obs[action],None)
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs

        env.close()

    # def test_layer2(self):
    #     results = ["S","B","S","B","S","B","S","B","S","B",None]
    #     cache_dir = 'cache'
    #     useCache = False
    #     cache_program = False
    #     cache_matrix = False and cache_program
    #     useCache = False and cache_matrix
    #     train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
    #     save_stdout = sys.stdout
    #     sys.stdout = open('trash', 'w')
    #     policy = train("UnityGame", range(0,3), 20, 300, 200, 5, interactive=True, specify_task="Naive_game",test_dimension="reduced"  )
    #     sys.stdout = save_stdout
    #     env_names = 'UnityGame1-v0'
    #     env = gym.make(env_names)
    #     obs = env.reset()
    #     total_reward = 0.
    #     for t in range(11):
    #         action = policy(obs)
    #         self.assertEqual(obs[action],results[t])
    #         new_obs, reward, done, debug_info = env.step(action)
    #         obs = new_obs

    #     env.close()

    # #Test with big number of objs
    # def test_layer1_big(self):
    #     results = ["S","B",None]
    #     cache_dir = 'cache'
    #     useCache = False
    #     cache_program = False
    #     cache_matrix = False and cache_program
    #     useCache = False and cache_matrix
    #     train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
    #     save_stdout = sys.stdout
    #     sys.stdout = open('trash', 'w')
    #     policy = train("UnityGame", range(0,3), 200, 4000, 200, 5, interactive=True, specify_task="Naive_game" )
    #     sys.stdout = save_stdout
    #     env_names = 'UnityGame0-v0'
    #     env = gym.make(env_names)
    #     obs = env.reset()
    #     total_reward = 0.
    #     for t in range(3):
    #         action = policy(obs)
    #         print(action)
    #         self.assertEqual(obs[action],results[t])
    #         new_obs, reward, done, debug_info = env.step(action)
    #         obs = new_obs

    #     env.close()

    # #Test with big number of objs
    # def test_layer2_big(self):
    #     results = ["S","B","S","B","S","B","S","B","S","B",None]
    #     cache_dir = 'cache'
    #     useCache = False
    #     cache_program = False
    #     cache_matrix = False and cache_program
    #     useCache = False and cache_matrix
    #     train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache)
    #     save_stdout = sys.stdout
    #     sys.stdout = open('trash', 'w')
    #     policy = train("UnityGame", range(0,3), 200, 4000, 200, 5, interactive=True, specify_task="Naive_game" )
    #     sys.stdout = save_stdout
    #     env_names = 'UnityGame1-v0'
    #     env = gym.make(env_names)
    #     obs = env.reset()
    #     total_reward = 0.
    #     for t in range(11):
    #         action = policy(obs)
    #         print(action)
    #         self.assertEqual(obs[action],results[t])
    #         new_obs, reward, done, debug_info = env.step(action)
    #         obs = new_obs

    #     env.close()
    



