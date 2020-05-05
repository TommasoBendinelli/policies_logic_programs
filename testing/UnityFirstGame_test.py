import unittest 
import pipeline
import gym
import numpy as np
import sys
import os


class TestLayers(unittest.TestCase):

    def test_layer1(self):
        results = ["S","B",None]
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w')
        policy = pipeline.train("UnityGame", range(0,3), 20, 300, 100, 5, interactive=True, specify_task="Naive_game" )
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
        results = ["S","B","S","B","S","B","S","B",None]
        sys.stdout = open(os.devnull, "w")
        policy = pipeline.train("UnityGame", range(0,3), 20, 300, 100, 5, interactive=True, specify_task="Naive_game" )
        sys.stdout = sys.__stdout__
        env_names = 'UnityGame0-v0'
        env = gym.make(env_names)
        obs = env.reset()
        total_reward = 0.
        for t in range(9):
            action = policy(obs)
            self.assertEqual(obs[action],results[t])
            new_obs, reward, done, debug_info = env.step(action)
            obs = new_obs

        env.close()
