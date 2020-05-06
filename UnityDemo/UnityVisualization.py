import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
import numpy as np
from UnityDemo.constants import *
import imageio

import matplotlib
matplotlib.use('TkAgg')
class UnityVisualization():
    fig_scale = 0.2

    def __init__(self, demonstration):
        self.curr_time = 0
        self.end_time = len(demonstration)-1
        self.dict_seq = list(zip(*demonstration))[0]
        self.action = list(zip(*demonstration))[1]
        self.height = len(self.dict_seq[0])
        self.width = len(self.dict_seq[0][0])
        self.drawings = []
        self.fig_scale = 0.2

    
    def visualize_demonstration(self):
        self.fig, self.ax = self.initialize_figure(self.height,self.width)
        self.render_onscreen()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        #self.fig.canvas.mpl_connect('pick_event', self.on_key)
        print(self.action[self.curr_time])
        plt.show()
    
    def save_demostration(self,video_out_path='out.mp4'):
        self.fig, self.ax = self.initialize_figure(self.height,self.width)
        self.start_recording_video(video_out_path)
        count = 0 
        while(True):
            print("Rendering frame num {}".format(count))
            curr = self.render()
            for i in range(8):
                self.recorded_video_frames.append(curr)    
            if self.action[self.curr_time] == (0,0):
                break
            self.to_next_timestep()
            count = count + 1
            for i in range(16):
                self.recorded_video_frames.append(curr)   
        imageio.mimsave(self.video_out_path, self.recorded_video_frames, 
            fps=8)
        print("Wrote out video to {}.".format(self.video_out_path))



    def to_next_timestep(self):
        if self.curr_time < self.end_time:
            self.curr_time = self.curr_time + 1
        else:
            print("No more timesteps")

    def on_key(self, event):
        self.to_next_timestep()
        print(self.action[self.curr_time])
        self.render_onscreen()
        self.fig.canvas.draw()
        return

    def start_recording_video(self, video_out_path):
        self.record_video = True
        self.recorded_video_frames = []
        self.video_out_path = video_out_path
    
    def render(self):
        return self.get_image(self.dict_seq[self.curr_time], self.action[self.curr_time], height = self.height, width = self.width)

    def render_onscreen(self):
        for drawing in self.drawings:
            drawing.remove()
        self.drawings = []

        for r in range(self.height):
            for c in range(self.width):
                # if r == 20 and c == 21:
                #     print("hello")
                token = self.dict_seq[self.curr_time][r][c]
                drawing = self.draw_token(token, r, c, self.ax, height = self.height,width=self.width)
                if drawing is not None:
                    self.drawings.append(drawing)
        
        if self.action[self.curr_time] != None:
            drawing = self.draw_action(self.action[self.curr_time],self.ax,height = self.height,width=self.width)
            if drawing is not None:
                        self.drawings.append(drawing)
    
    @classmethod
    def get_image(cls, observation, action, mode='human', close=False, height = None, width = None):
        height, width = observation.shape

        fig, ax = cls.initialize_figure(height, width)

        for r in range(height):
            for c in range(width):
                token = observation[r, c]
                cls.draw_token(token, r, c, ax, height, width)

        if action is not None:
            cls.draw_action(action, ax, height, width)

        im = cls.fig2data(fig)
        plt.close(fig)

        return im

    @staticmethod
    def fig2data(fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    @classmethod
    def draw_action(cls, action, ax, height, width):
        r, c = action
        if not (isinstance(r, int) or isinstance(r, np.int8) or isinstance(r, np.int64)):
            r -= 0.5
            c -= 0.5
        oi = OffsetImage(TOKEN_IMAGES[HAND_ICON], zoom = 1 * cls.fig_scale * (2.5 / max(height, width)**0.5))
        box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)
        ax.add_artist(box)
        return box

    @classmethod
    def draw_token(cls,token, r, c, ax, height, width):
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
            oi = OffsetImage(im, zoom = 1 * (cls.fig_scale*2  / max(height, width)**0.5))
            box = AnnotationBbox(oi, (c + 0.5, (height - 1 - r) + 0.5), frameon=False)

            ax.add_artist(box)
            return box

    # @staticmethod
    # def box_creator(string):
    #     for i in range(string.replace(" ","")):
    #         yield i


    @classmethod
    def initialize_figure(cls,height, width):
        fig = plt.figure(figsize=((width + 2) * cls.fig_scale  , (height) * cls.fig_scale  + 2 ))
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