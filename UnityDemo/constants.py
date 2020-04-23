import matplotlib.pyplot as plt
from generalization_grid_games.envs.utils import get_asset_path

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