import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, FancyArrow
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import RegularPolygon, FancyArrow
from generalization_grid_games.envs.utils import get_asset_path
import numpy as np

class UnityVisualization():
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

    def __init__(self,action, dict_seq):
        self.curr_time = 0
        self.end_time = len(dict_seq)-1
        self.dict_seq = dict_seq
        self.action = action
        self.height = len(dict_seq[0])
        self.width = len(dict_seq[0][0])
        self.drawings = []
        self.fig_scale = 0.2
    
    def visualize_demonstration(self):
        self.fig, self.ax = self.initialize_figure(self.height,self.width)
        self.render_onscreen()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        #self.fig.canvas.mpl_connect('pick_event', self.on_key)
        print(self.action[self.curr_time])
        plt.show()

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
                # if r == 20 and c == 21:
                #     print("hello")
                token = self.dict_seq[self.curr_time][r][c]
                drawing = self.draw_token(token, r, c, self.ax, height = self.height,width=self.width)
                if drawing is not None:
                    self.drawings.append(drawing)

    def draw_token(self,token, r, c, ax, height, width):
        if token in self.EMPTY:
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
            im = self.TOKEN_IMAGES[tk]
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