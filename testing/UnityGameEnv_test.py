import unittest 
import pipeline
import gym
import numpy as np
import sys
import os
import time

class UnityGameEnv_test(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s: %.3f' % (self.id(), t))

    def test_integration(self):
        env_names = 'UnityGame0-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        results = ["S","highlighted","S"]
        action = [(19,22),(19,22),(19,22)]
        for t in range(3):
            self.assertEqual(obs[action[t]],results[t])
            new_obs, reward, done, debug_info = env.step(action[t])
            obs = new_obs
        # cache_dir = 'cache'
        # useCache = False
        # cache_program = False
        # cache_matrix = False and cache_program
        # useCache = False and cache_matrix
        # train = pipeline.pipeline_manager(cache_dir,cache_program,cache_matrix,useCache, is_logging_enabled=False)
        # #save_stdout = sys.stdout
        # #sys.stdout = open('trash', 'w')
        # policy = train("UnityGame", range(0,3), 20, 200, 300, 5, interactive=True, specify_task="Game0", test_dimension="reduced" )
        # #sys.stdout = save_stdout
        # env_names = 'UnityGame0-v0'
        # env = gym.make(env_names)
        # obs = env.reset()
        # total_reward = 0.
        # for t in range(3):
        #     action = policy(obs)
        #     self.assertEqual(obs[action],results[t])
        #     new_obs, reward, done, debug_info = env.step(action)
        #     obs = new_obs

        # env.close()