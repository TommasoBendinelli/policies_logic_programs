import json 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
#from .generalization_grid_game import create_gym_envs
#from generalization_grid_games.envs.playingXYZGeneralizationGridGame import PlayingXYZGeneralizationGridGame
from generalization_grid_games.envs.utils import get_asset_path, changeResolution
from copy import deepcopy
from UnityDemo.UnityVisualization import UnityVisualization
from UnityDemo.GetDemonstration import DemonstrationHandler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import sys
from PIL import Image

import numpy as np
import os 


EMPTY = [None,""]
P = 'P'
P_Clicked = 'P_highlighted'
S = 'S'
S_CLicked = 'S_highlighted'
B = 'B'
START = 's'
PASS = 'pass'
CLICK = 'click'
ALL_TOKENS = [EMPTY, P, S, B, PASS, START]
ALL_ACTION_TOKENS = [CLICK, PASS]

TOKEN_IMAGES = {
    P: plt.imread(get_asset_path('p.png')),
    S: plt.imread(get_asset_path('s.png')),
    B: plt.imread(get_asset_path('b.png')),
    START: plt.imread(get_asset_path('start.png')),
    P_Clicked: plt.imread(get_asset_path('P_highlighted.png')),
    S_CLicked: plt.imread(get_asset_path('S_highlighted.png'))
}


def demonstration_as_dict_creator(data):
    demonstration = list()
    for time_step in range(len(data)):
        data[time_step] = converter(data[time_step])
        demonstration.append(data[time_step])
    return demonstration

#Convert "" to None
def converter(demo):
    for x in range(len(demo)):
        for y in range(len(demo[0])):
            if demo[x][y] == "":
                demo[x][y] = None
    return demo

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



demo_object = DemonstrationHandler("demo2.json",reversed_demo=True)
final_demo = demo_object()
# final_demo = demo_object(raw=True)
# final_demo = list(zip(*final_demo))[0]
# demo_object.other_representation(final_demo)
interaction = UnityVisualization(final_demo)
interaction.visualize_demonstration()



