import json 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
#from .generalization_grid_game import create_gym_envs
#from generalization_grid_games.envs.playingXYZGeneralizationGridGame import PlayingXYZGeneralizationGridGame
#from generalization_grid_games.envs.utils import get_asset_path, changeResolution
from copy import deepcopy
from UnityDemo.UnityVisualization import UnityVisualization
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import sys
from PIL import Image
from UnityDemo.constants import *
import time
import random
from itertools import product
import numpy as np
import os 
import math
from collections import defaultdict

class DemonstrationHandler():

    def __init__(self,path,false_demonstration_per_step=250,complete_random=False,reversed_demo=False):
        with open('unity_demonstrations/' + path) as json_file:
            data = json.load(json_file)
        dict_seq = self.demonstration_as_dict_creator(data)
        self.dict_seq = [(np.array(dict_seq[x]), None) for x in range(len(dict_seq))]
        action, final_demo = self.find_difference(dict_seq)
        self.initial_layer = np.array(final_demo[0])
        self.reversed = reversed_demo
        action_shifterd = self.rotate(action,-1)
        self.demonstrations_reversed = [(np.array(final_demo[x]), action_shifterd[x]) for x in reversed(range(len(final_demo)))]
        if self.reversed == True:
            self.demonstrations = self.demonstrations_reversed
        else:
            self.demonstrations = [(np.array(final_demo[x]), action[x]) for x in range(len(final_demo))]
        self.false_demonstration_per_step = false_demonstration_per_step
        self.complete_random = complete_random
        #self.final_demo = np.array(final_demo)
        #self.action = np.array(action)

    def __call__(self,only_initial_layer=False,raw=False):
        if only_initial_layer:
            return self.initial_layer
        if raw:
            return self.dict_seq
        return self.demonstrations 
    
    @staticmethod
    def rotate(l, n):
        return l[n:] + l[:n]

    @staticmethod
    def other_representation(demonstration_seq):
        delta_dict = defaultdict(lambda: [(0,0),(0,0)]) 
        register_movement = defaultdict(lambda: defaultdict(lambda: [(0,0),(0,0)]) ) 
        for step in range(1,len(demonstration_seq)):
            flag = 0
            for x in range(len(demonstration_seq[0])):
                for y in range(len(demonstration_seq[0][0])):
                    if (demonstration_seq[step][x][y] != demonstration_seq[step-1][x][y]):
                        if demonstration_seq[step-1][x][y] != None and demonstration_seq[step-1][x][y] != "B" :
                            curr = demonstration_seq[step-1][x][y]
                            register_movement[step][curr][0] = (x,y) 
                            flag = 1
                        if demonstration_seq[step][x][y] != None and demonstration_seq[step][x][y] != "B" :
                            curr = demonstration_seq[step][x][y]
                            register_movement[step][curr][1] = (x,y)
                            flag = 1
        return register_movement
    
    @staticmethod
    def robust_transition(demonstration_seq,other_representation_transition):
        if not isinstance(other_representation_transition, defaultdict):
            print("ERROR")
            return
        curr = other_representation_transition 
        for delta in other_representation_transition.keys():
            if delta.keys() > 1:
              print("Hello")  
    # @staticmethod
    # def return_biggest_distance(p1,p2):
    #     distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )






    @staticmethod
    def find_difference(demonstration_seq):
        final_demo = deepcopy(demonstration_seq)
        action = []
        counter = 0 
        #diff = [key:[[None]*len(demonstration_seq[0][0]) for x in range(len(demonstration_seq[0]))] for key in demonstration_seq.keys()}
        for step in range(1,len(demonstration_seq)):
            all_equal = 0
            flag = 0
            for x in range(len(demonstration_seq[0])):
                for y in range(len(demonstration_seq[0][0])):
                    # if x == 35 and y == 3:
                    #     print("Hello")
                    # if x == 31 and y == 22:
                    #     print("Hello2")
                    if (demonstration_seq[step][x][y] == demonstration_seq[step-1][x][y]):
                        all_equal = all_equal + 1
                    elif demonstration_seq[step-1][x][y] == P:
                        final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                        final_demo[step+counter][x][y] =  "P_highlighted"
                        action.append((x,y))
                        if flag == 1:
                            action[-1], action[-2] = action[-2], action[-1]
                        counter = counter + 1
                    elif demonstration_seq[step-1][x][y] == S:
                        final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                        final_demo[step+counter][x][y] =  "S_highlighted"
                        action.append((x,y))
                        if flag == 1:
                            action[-1], action[-2] = action[-2], action[-1]
                        counter = counter + 1
                    elif demonstration_seq[step-1][x][y] in OBJECTS:
                        final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                        final_demo[step+counter][x][y] =  demonstration_seq[step-1][x][y] + "_highlighted"
                        action.append((x,y))
                        if flag == 1:
                            action[-1], action[-2] = action[-2], action[-1]
                        counter = counter + 1
                    else:
                        action.append((x,y))
                        flag = 1
                        #print("what is going on?")
                        #print("I do not know")
            print("Total number of squares {}: current {}".format(len(demonstration_seq[0][0])*len(demonstration_seq[0]),all_equal))
            #if all_equal == len(demonstration_seq[0][0])*len(demonstration_seq[0]):
        action.append((0,0))

        return action, final_demo
    
    @classmethod
    def demonstration_as_dict_creator(cls,data):
        demonstration = list()
        for time_step in range(len(data)):
            data[time_step] = cls.to_none_converter(data[time_step])
            data[time_step] = cls.to_multiple_converter(data[time_step])
            demonstration.append(data[time_step])
        return demonstration

    @staticmethod
    def to_none_converter(demo):
        for x in range(len(demo)):
            for y in range(len(demo[0])):
                if demo[x][y] == "":
                    demo[x][y] = None
        return demo

    @staticmethod
    def to_multiple_converter(demo):
        for x in range(len(demo)):
            for y in range(len(demo[0])):
                if isinstance(demo[x][y],str):
                    demo[x][y] = demo[x][y].split()[-1]
        return demo
    
    def extract_examples_from_demonstration(self):
        positive_examples = []
        negative_examples = []
        start = time.time()
        for idx, demonstration_item in enumerate(self.demonstrations):
            #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item_TEST(idx,demonstration)
            #demo_positive_examples, demo_negative_examples = cls.extract_examples_from_demonstration_item(demonstration_item)
            demo_positive_examples, demo_negative_examples = self.extract_examples_from_demonstration_item_sample(demonstration_item,self.false_demonstration_per_step,self.complete_random)
            positive_examples.extend(demo_positive_examples)
            negative_examples.extend(demo_negative_examples)
        for idx, demonstration_item in enumerate(self.demonstrations_reversed):
            state, loc = demonstration_item
            negative_examples.extend([(state, loc)])
        
        print("Total time for generating demonstrations: {}".format(time.time()-start))
        return positive_examples, negative_examples
    

    @staticmethod
    def extract_examples_from_demonstration_item_sample(demonstration_item,false_demonstration_per_step, complete_random):
        state, loc = demonstration_item

        positive_examples = [(state, loc)]
        negative_examples = []

        if not any([x in state for x in PICKED_UP]):
            for elem in TERRAIN:
                indeces = np.array(np.where(state==elem))
                rand_vec = np.random.choice(indeces.shape[1],50)
                for rand in rand_vec:
                    nums = (int(indeces[0,rand]), int(indeces[1,rand]))
                    if nums != (0,0):
                        negative_examples.append((state, nums))
        
        if  any([x in state for x in PICKED_UP]):
            for elem in OBJECTS:
                indeces = np.array(np.where(state==elem))
                if np.array(indeces).shape[1] != 0:
                    rand_vec = np.random.choice(indeces.shape[1],20)
                    for rand in rand_vec:
                        nums = (int(indeces[0,rand]), int(indeces[1,rand]))
                        if nums != (0,0):
                            negative_examples.append((state, nums))

        if complete_random == False:
            #Start by considering all actions on objects
            for r in range(state.shape[0]):
                for c in range(state.shape[1]):
                    if (r, c) == loc or state[r,c] == None:
                        continue
                    else:
                        negative_examples.append((state, (r, c)))
        
        try:
            nums = (0,0) 
            if nums != loc:
                negative_examples.append((state, nums))
            #Sample 250 random demonstrations instead of computing all of them
            indeces = random.sample(list(product(range(state.shape[0]),range(state.shape[1]))),false_demonstration_per_step)

            for nums in indeces:
                if (state,nums) in negative_examples:
                    continue
                if nums == loc:
                    continue
                else:
                    negative_examples.append((state, nums))
            return positive_examples, negative_examples
        except:
            print("Skipped as no so many demonstrations")
            for r in range(state.shape[0]):
                for c in range(state.shape[1]):
                        if (r, c) == loc:
                            continue
                        else:
                            negative_examples.append((state, (r, c)))

            return positive_examples, negative_examples
    
def get_demonstrations_name(demo_path = "unity_demonstrations"):
    return os.listdir(demo_path) 