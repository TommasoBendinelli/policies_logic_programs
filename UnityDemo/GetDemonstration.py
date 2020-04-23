import json 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
#from .generalization_grid_game import create_gym_envs
from generalization_grid_games.envs.playingXYZGeneralizationGridGame import PlayingXYZGeneralizationGridGame
#from generalization_grid_games.envs.utils import get_asset_path, changeResolution
from copy import deepcopy
from UnityDemo.UnityVisualization import UnityVisualization
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import sys
from PIL import Image
from UnityDemo.constants import *

import numpy as np
import os 

class GetDemonstration():

    def __init__(self):
        with open('unity_demonstrations/matrix.json') as json_file:
            data = json.load(json_file)
        dict_seq = self.demonstration_as_dict_creator(data)
        self.action, self.final_demo = self.find_difference(dict_seq)

    def __call__(self):
        return self.action, self.final_demo

    @staticmethod
    def find_difference(demonstration_seq):
        final_demo = deepcopy(demonstration_seq)
        action = []
        counter = 0 
        #diff = [key:[[None]*len(demonstration_seq[0][0]) for x in range(len(demonstration_seq[0]))] for key in demonstration_seq.keys()}
        for step in range(1,len(demonstration_seq)):
            all_equal = 0
            for x in range(len(demonstration_seq[0])):
                for y in range(len(demonstration_seq[0][0])):
                    if (demonstration_seq[step][x][y] == demonstration_seq[step-1][x][y]):
                        all_equal = all_equal + 1
                    elif demonstration_seq[step-1][x][y] == P:
                        final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                        final_demo[step+counter][x][y] =  "P_highlighted"
                        action.append(([x],[y]))
                        counter = counter + 1
                    elif demonstration_seq[step-1][x][y] == S:
                        final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                        final_demo[step+counter][x][y] =  "S_highlighted"
                        action.append(([x],[y]))
                        counter = counter + 1
                    else:
                        action.append(([x],[y]))
                        #print("what is going on?")
                        #print("I do not know")
            
            #if all_equal == len(demonstration_seq[0][0])*len(demonstration_seq[0]):
        action.append((0,0))

        return action, final_demo
    
    @classmethod
    def demonstration_as_dict_creator(cls,data):
        demonstration = list()
        for time_step in range(len(data)):
            data[time_step] = cls.to_none_converter(data[time_step])
            demonstration.append(data[time_step])
        return demonstration

    @staticmethod
    def to_none_converter(demo):
        for x in range(len(demo)):
            for y in range(len(demo[0])):
                if demo[x][y] == "":
                    demo[x][y] = None
        return demo