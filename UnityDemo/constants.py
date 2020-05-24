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
CUBE_RED = "CR"
CUBE_RED_Clicked = "CR_highlighted"
CUBE_BLUE = "CB"
CUBE_BLUE_Clicked = "CB_highlighted"
CUBE_GREEN = "CG"
CUBE_GREEN_Clicked = "CG_highlighted"
CUBE_BLACK = "CBLA"
CUBE_BLACK_Clicked = "CBLA_highlighted"
CUBE_BROWN = "CBR"
CUBE_BROWN_Clicked = "CBR_highlighted"
CUBE_PINK = "CP"
CUBE_PINK_Clicked = "CP_highlighted"
CUBE_YELLOW = "CY"
CUBE_YELLOW_Clicked = "CY_highlighted"
CUBE_GREY = "CGR" 
CUBE_GREY_Clicked = "CGR_highlighted"
Clicked = "highlighted"
PICKED_UP = [Clicked] #[S_CLicked, P_Clicked,CUBE_RED_Clicked,CUBE_BLUE_Clicked,CUBE_BLACK_Clicked,CUBE_PINK_Clicked,CUBE_YELLOW_Clicked,CUBE_GREY_Clicked]
TERRAIN = [None, B]
OBJECTS = [S,P,CUBE_RED,CUBE_BLUE,CUBE_GREEN,CUBE_BLACK,CUBE_PINK,CUBE_YELLOW,CUBE_GREY]

HAND_ICON = 'hand'
ALL_TOKENS = [EMPTY, P, S, B, PASS, START]
#ALL_ACTION_TOKENS = [CLICK]

TOKEN_IMAGES = {
    Clicked: plt.imread(get_asset_path("star.png")),
    P: plt.imread(get_asset_path('P.png')),
    S: plt.imread(get_asset_path('S.png')),
    B: plt.imread(get_asset_path('B.png')),
    START: plt.imread(get_asset_path('start.png')),
    P_Clicked: plt.imread(get_asset_path('P_highlighted.png')),
    S_CLicked: plt.imread(get_asset_path('S_highlighted.png')),
    HAND_ICON: plt.imread(get_asset_path('blue_hand_icon.png')),
    CUBE_RED: plt.imread(get_asset_path('cube_red.png')),
    CUBE_BLUE: plt.imread(get_asset_path('cube_blue.png')),
    CUBE_GREEN: plt.imread(get_asset_path('cube_green.png')),
    CUBE_BLACK: plt.imread(get_asset_path('cube_black.png')),
    CUBE_YELLOW: plt.imread(get_asset_path('cube_yellow.png')),
    CUBE_PINK: plt.imread(get_asset_path('cube_pink.png')),
    CUBE_GREY: plt.imread(get_asset_path('cube_grey.png')),
    CUBE_BROWN: plt.imread(get_asset_path('cube_brown.png')),
    CUBE_BLUE_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_BLACK_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_RED_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_GREEN_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_RED_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_BROWN_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_PINK_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_GREY_Clicked: plt.imread(get_asset_path('cube_selected.png')),
    CUBE_YELLOW_Clicked: plt.imread(get_asset_path('cube_selected.png'))
}