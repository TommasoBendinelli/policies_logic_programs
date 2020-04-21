import json 
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
#from .generalization_grid_game import create_gym_envs
from generalization_grid_games.envs.playingXYZGeneralizationGridGame import PlayingXYZGeneralizationGridGame
from generalization_grid_games.envs.utils import get_asset_path, changeResolution
from copy import deepcopy
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
        demonstration.append(data[time_step])
    return demonstration


def find_difference(demonstration_seq):
    final_demo = deepcopy(demonstration_seq)
    counter = 0 
    #diff = [key:[[None]*len(demonstration_seq[0][0]) for x in range(len(demonstration_seq[0]))] for key in demonstration_seq.keys()}
    for step in range(1,len(demonstration_seq)):
        for x in range(len(demonstration_seq[0])):
            for y in range(len(demonstration_seq[0][0])):
                if (demonstration_seq[step][x][y] == demonstration_seq[step-1][x][y]):
                    continue
                elif demonstration_seq[step-1][x][y] == P:
                    final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                    final_demo[step+counter][x][y] =  "P_highlighted"
                    counter = counter + 1
                elif demonstration_seq[step-1][x][y] == S:
                    final_demo.insert(step+counter,deepcopy(final_demo[step-1+counter]))
                    final_demo[step+counter][x][y] =  "S_highlighted"
                    counter = counter + 1
    return final_demo





# def visualize_demonstration(dict_seq):
#     fig, ax = initialize_figure(len(dict_seq[0]),len(dict_seq[0]))
#     populate_fig(dict_seq[0], ax)
#     fig.canvas.mpl_connect('key_press_event', interaction.on_key)
#     plt.show()

# def populate_fig(matrix, ax):
#     height = len(matrix)
#     width = len(matrix[0])
#     for r in range(height):
#         for c in range(width):
#             token = matrix[r][c]
#             draw_token(token, r, c, ax, width=width, height = height)

class Visualization():
    def __init__(self,dict_seq):
        self.curr_time = 0
        self.end_time = len(dict_seq)-1
        self.dict_seq = dict_seq
        self.height = len(dict_seq[0])
        self.width = len(dict_seq[0][0])
        self.drawings = []
        self.fig_scale = 0.2
    
    def visualize_demonstration(self):
        self.fig, self.ax = self.initialize_figure(self.height,self.width)
        self.render_onscreen()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        #self.fig.canvas.mpl_connect('pick_event', self.on_key)
        plt.show()

    def to_next_timestep(self):
        if self.curr_time < self.end_time:
            self.curr_time = self.curr_time + 1
        else:
            print("No more timesteps")

    def on_key(self, event):
        self.to_next_timestep()
        print("here")
        self.render_onscreen()
        self.fig.canvas.draw()
        return
        #plt.show()

    # def render_onscreen_t(self):
    #     for drawing in self.drawings:
    #         drawing.remove()
    #     self.drawings = []

    #     for r in range(self.height):
    #         for c in range(self.width):
    #             token = self.dict_seq[self.curr_time][r][c]
    #             drawing = self.draw_token(token, r, c, self.ax, height = self.height,width=self.width)
    #             if drawing is not None:
    #                 self.drawings.append(drawing)
    
    # @staticmethod
    # def merge_multiple_images(token):
    #     images = [TOKEN_IMAGES[tk] for tk in list(token.replace(" ",""))]
    #     widths, heights = zip(*(i.shape for i in images))

    #     total_width = sum(widths)
    #     max_height = max(heights)

    #     new_im = Image.new('RGB', (total_width, max_height))

    #     x_offset = 0
    #     for im in images:
    #         new_im.paste(Image.fromarray(im[:,:,:3]), (x_offset,0))
    #         x_offset += im.size[0]

    #     new_im.save('test.jpg')



    def render_onscreen(self):
        for drawing in self.drawings:
            drawing.remove()
        self.drawings = []

        for r in range(self.height):
            for c in range(self.width):
                if r == 20 and c == 21:
                    print("hello")
                token = self.dict_seq[self.curr_time][r][c]
                drawing = self.draw_token(token, r, c, self.ax, height = self.height,width=self.width)
                if drawing is not None:
                    self.drawings.append(drawing)

    def draw_token(self,token, r, c, ax, height, width):
        if token in EMPTY:
            edge_color = '#888888'
            face_color = 'white'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

            return drawing

        else:
            edge_color = '#888888'
            face_color = '#DDDDDD'
            
            drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
                                         numVertices=4,
                                         radius=0.5 * np.sqrt(2),
                                         orientation=np.pi / 4,
                                         ec=edge_color,
                                         fc=face_color)
            ax.add_patch(drawing)

            tk = token.split()[-1]
            im = TOKEN_IMAGES[tk]
            oi = OffsetImage(im, zoom = 1 * (self.fig_scale*2  / max(height, width)**0.5))
            box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

            ax.add_artist(box)
            return box

    # @staticmethod
    # def box_creator(string):
    #     for i in range(string.replace(" ","")):
    #         yield i


    def initialize_figure(self,height, width):
        fig = plt.figure(figsize=((width + 2) * self.fig_scale  , (height) * self.fig_scale  + 2 ))
        ax = fig.add_axes((0.05, 0.1, 0.9, 0.9),
                                    aspect='equal', frameon=False,
                                    xlim=(-0.05, width + 0.05),
                                    ylim=(-0.05, height + 0.05))
        ax.set_picker(True)
        ax.name = "Grid"
        #axbox = fig.add_axes([0.1, 0.02, 0.8, 0.075], xlim=(-0.05, width + 0.05),
        #                            ylim=(height + 0.05, height + 0.10))
        #axbox.set_picker(True)
        #axbox.name = "TextBox"
        #text_box = TextBox(axbox,"", initial=" ")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(plt.NullFormatter())
            axis.set_major_locator(plt.NullLocator())

        return fig, ax


# def concat_images(imga, imgb):
#     """
#     Combines two color image ndarrays side-by-side.
#     """
#     ha,wa = imga.shape[:2]
#     hb,wb = imgb.shape[:2]
#     max_height = np.max([ha, hb])
#     total_width = wa+wb
#     new_img = np.zeros(shape=(max_height, total_width, 3))
#     new_img[:ha,:wa]=imga
#     new_img[:hb,wa:wa+wb]=imgb
#     return new_img

# def concat_n_images(image_path_list):
#     """
#     Combines N color images from a list of image paths.
#     """
#     images = list(image_path_list.replace(" ",""))
#     output = None
#     for i, image_key in enumerate(images):
#         img = TOKEN_IMAGES[image_key]
#         if i==0:
#             output = img
#         else:
#             output = concat_images(output, img)
#     return output

with open('unity_demonstrations/matrix.json') as json_file:
    data = json.load(json_file)

dict_seq = demonstration_as_dict_creator(data)
final_demo = find_difference(dict_seq)
interaction = Visualization(final_demo)
interaction.visualize_demonstration()
#output = concat_n_images("B P")
#plt.imshow(output)
#plt.show()
# 
# 
# interaction.visualize_demonstration()
# # for keys in dict_seq.keys():
#     print(dict_seq[keys])
# diff = find_difference(dict_seq)
#print(diff)

#visualize_demonstration(dict_seq)




