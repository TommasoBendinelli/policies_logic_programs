import unittest 
import pipeline
import gym
import numpy as np
import sys
import os
import time
from grammar_utils import maximum_number_of_program
from env_settings import get_object_types
from dsl import create_grammar_unity

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
    
    def test_total_number(self):
        base_class_name = "UnityGame_reduced"
        object_types = get_object_types(base_class_name)
        object_types = object_types 
        grammar = create_grammar_unity(object_types) 
        num = maximum_number_of_program(grammar)
        self.assertEqual(num,257)
        base_class_name = "UnityGame"
        object_types = get_object_types(base_class_name)
        object_types = object_types 
        grammar = create_grammar_unity(object_types) 
        num = maximum_number_of_program(grammar)
        self.assertEqual(num,2113)