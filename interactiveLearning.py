# import matplotlib
# matplotlib.use('TkAgg')
# from generalization_grid_games.envs.utils import fig2data,get_asset_path, changeResolution

# import numpy as np
# import matplotlib.pyplot as plt
# import generalization_grid_games 
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from matplotlib.widgets import TextBox
# from generalization_grid_games.envs.generalization_grid_game import PlayingXYZGeneralizationGridGame
# #from globals_var import action, action_done
# from matplotlib.patches import RegularPolygon, FancyArrow
# import os

# EMPTY = 'empty'
# X = 'x'
# Y = 'y'
# Z = 'z'
# PASS = 'pass'
# ALL_TOKENS = [EMPTY, X, Y, Z, PASS]
# ALL_ACTION_TOKENS = [X, Y, Z, PASS]

# #Change image resolution
# for image in os.listdir(get_asset_path('raw/')):
#     if image == ".DS_Store":
#         continue
#     path = get_asset_path('raw/'+image)
#     changeResolution(path,get_asset_path('') + image)

# ALL_TOKENS = [EMPTY, X, Y, Z, PASS]
# TOKEN_IMAGES = {
#     X: plt.imread(get_asset_path('x.png')),
#     Y: plt.imread(get_asset_path('y.png')),
#     Z: plt.imread(get_asset_path('z.png'))
# }
# class InteractiveLearning(PlayingXYZGeneralizationGridGame):
#     def __init__(self,layer_0):
#         self.layout = np.array(layer_0, dtype=object)
#         self.current_layout = self.layout
#         self.initial_layout = self.layout.copy()
#         self.action_lock = False

#         # Create the figure and axes
#         self.height, self.width = self.layout.shape
#         height, width = self.layout.shape
#         self.fig, self.ax, self.textbox = self.initialize_figure(height, width)
#         self.drawings = []
#         self.render_onscreen()
#         self.action = None
#         self.fig.canvas.mpl_connect('pick_event', self.button_press)
            
#         # Create event for keyboard
#         self.textbox.on_submit(self.submit)
#         self.current_text_value = None

#         plt.show()



#     def button_press(self, event):
#         if event.artist.name == "Grid" and self.current_text_value != None:
#             event = event.mouseevent
#             if self.action_lock:
#                 return
#             if (event.xdata is None) or (event.ydata is None):
#                 return
#             i, j = map(int, (event.xdata, event.ydata))
        
#             if (i < 0 or j < 0 or i >= self.width or j >= self.height):
#                 return

#             self.action_lock = True
#             c, r = i, self.height - 1 - j
#             if event.button == 1:
#                 action_done = 1
#                 self.action = (self.current_text_value,(r, c))
#                 self.step(action)
#                 self.fig.canvas.draw()
#             # self.action_lock = False
            
#         else: 
#             return

#     # def render_onscreen(self):
#     #     for drawing in self.drawings:
#     #         drawing.remove()
#     #     self.drawings = []

#     #     for r in range(self.height):
#     #         for c in range(self.width):
#     #             token = self.current_layout[r, c]
#     #             drawing = self.draw_token(token, r, c, self.ax, self.height, self.width, token_scale=0.75)
#     #             if drawing is not None:
#     #                 self.drawings.append(drawing)

#     ### Helper stateless methods
#     @classmethod
#     def initialize_figure(cls, height, width):
#         fig = plt.figure(figsize=((width + 2) * cls.fig_scale , (height) * cls.fig_scale + 2 ))
#         ax = fig.add_axes((0.05, 0.1, 0.9, 0.9),
#                                     aspect='equal', frameon=False,
#                                     xlim=(-0.05, width + 0.05),
#                                     ylim=(-0.05, height + 0.05))
#         ax.set_picker(True)
#         ax.name = "Grid"
#         axbox = fig.add_axes([0.1, 0.02, 0.8, 0.075], xlim=(-0.05, width + 0.05),
#                                     ylim=(height + 0.05, height + 0.10))
#         axbox.set_picker(True)
#         axbox.name = "TextBox"
#         text_box = TextBox(axbox,"", initial="Insert x, y, z ")
#         for axis in (ax.xaxis, ax.yaxis):
#             axis.set_major_formatter(plt.NullFormatter())
#             axis.set_major_locator(plt.NullLocator())

#         return fig, ax, text_box
        
#     @classmethod
#     def get_image(cls, observation, action, mode='human', close=False):
#         height, width = observation.shape

#         fig, ax = cls.initialize_figure(height, width)

#         for r in range(height):
#             for c in range(width):
#                 token = observation[r, c]
#                 cls.draw_token(token, r, c, ax, height, width)

#         if action is not None:
#             cls.draw_action(action, ax, height, width)

#         im = fig2data(fig)
#         plt.close(fig)

#         return im

#     # @classmethod
#     # def initialize_figure(cls, height, width, a):
#     #     fig = plt.figure(figsize=((width + 2) * cls.fig_scale, (height + 2) * cls.fig_scale))
#     #     ax = fig.add_axes((0.05, 0.05, 0.9, 0.9),
#     #                                 aspect='equal', frameon=False,
#     #                                 xlim=(-0.05, width + 0.05),
#     #                                 ylim=(-0.05, height + 0.05))
#     #     for axis in (ax.xaxis, ax.yaxis):
#     #         axis.set_major_formatter(plt.NullFormatter())
#     #         axis.set_major_locator(plt.NullLocator())
#     #     return fig, ax

#     # @classmethod
#     # def draw_action(cls, action, ax, height, width):
#     #     r, c = action
#     #     if not (isinstance(r, int) or isinstance(r, np.int8) or isinstance(r, np.int64)):
#     #         r -= 0.5
#     #         c -= 0.5
#     #     oi = OffsetImage(cls.hand_icon, zoom = 0.3 * cls.fig_scale * (2.5 / max(height, width)**0.5))
#     #     box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)
#     #     ax.add_artist(box)

#     #     return fig, ax, text_box


#     @classmethod
#     def draw_token(cls, token, r, c, ax, height, width, token_scale=1.0):
#         if token == EMPTY:
#             edge_color = '#888888'
#             face_color = 'white'
            
#             drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
#                                          numVertices=4,
#                                          radius=0.5 * np.sqrt(2),
#                                          orientation=np.pi / 4,
#                                          ec=edge_color,
#                                          fc=face_color)
#             ax.add_patch(drawing)

#             return drawing

#         else:
#             edge_color = '#888888'
#             face_color = '#DDDDDD'
            
#             drawing = RegularPolygon((c + 0.5, (height - 1 - r) + 0.5),
#                                          numVertices=4,
#                                          radius=0.5 * np.sqrt(2),
#                                          orientation=np.pi / 4,
#                                          ec=edge_color,
#                                          fc=face_color)
#             ax.add_patch(drawing)

#             im = TOKEN_IMAGES[token]
#             oi = OffsetImage(im, zoom = cls.fig_scale * (token_scale / max(height, width)**0.5))
#             box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

#             ax.add_artist(box)

#     def submit(self, text_at_submit):
#         if str.split(text_at_submit) and (str.split(text_at_submit)[0] in ('xyz') or text_at_submit.strip() == "pass"):
#             self.current_text_value = str.split(text_at_submit)[0]
#         else: self.current_text_value = None 

#     def transition(self,layout, action):
#         cval, pos  = action #i.e. (x, (3,1))
#         r, c = pos 
#         height, width = layout.shape
#         new_layout = layout.copy()
#         token = layout[r, c]
#         #cval = self.current_text_value
#         if cval == X or cval == Y or cval == Z:
#             return InteractiveLearning.add(new_layout,cval,r, c)
#         else: return new_layout
    
#     @staticmethod
#     def add(layout, token, r, c):
#         layout[r,c] = token
#         return layout