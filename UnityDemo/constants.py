import matplotlib.pyplot as plt
from generalization_grid_games.envs.utils import get_asset_path

EMPTY = [None,""]
P = 'P'
P_Clicked = 'P_highlighted'
S = 'S'
S_CLicked = 'S_highlighted'
B = 'B'
START = 'start'
PASS = 'pass'
PICKED_UP = [S_CLicked, P_Clicked]
TERRAIN = [None, B]
OBJECTS = [S,P]

HAND_ICON = 'hand'
ALL_TOKENS = [EMPTY, P, S, B, PASS, START]
#ALL_ACTION_TOKENS = [CLICK]

TOKEN_IMAGES = {
    P: plt.imread(get_asset_path('p.png')),
    S: plt.imread(get_asset_path('s.png')),
    B: plt.imread(get_asset_path('b.png')),
    START: plt.imread(get_asset_path('start.png')),
    P_Clicked: plt.imread(get_asset_path('P_highlighted.png')),
    S_CLicked: plt.imread(get_asset_path('S_highlighted.png')),
    HAND_ICON: plt.imread(get_asset_path('blue_hand_icon.png'))
}