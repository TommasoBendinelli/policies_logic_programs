from generalization_grid_games.envs import two_pile_nim as tpn
from generalization_grid_games.envs import checkmate_tactic as ct 
from generalization_grid_games.envs import stop_the_fall as stf
from generalization_grid_games.envs import chase as ec
from generalization_grid_games.envs import reach_for_the_star as rfts
from generalization_grid_games.envs import playing_with_XYZ as xyz
from UnityDemo import constants as unity

import generalization_grid_games


def get_object_types(base_class_name):
    if base_class_name == 'TwoPileNim':
        return ('tpn.EMPTY', 'tpn.TOKEN', 'None')
    if base_class_name == 'CheckmateTactic':
        return ('ct.EMPTY', 'ct.HIGHLIGHTED_WHITE_QUEEN', 'ct.BLACK_KING', 'ct.HIGHLIGHTED_WHITE_KING', 'ct.WHITE_KING', 'ct.WHITE_QUEEN', 'None')
    if base_class_name == 'StopTheFall':
        return ('stf.EMPTY', 'stf.FALLING', 'stf.RED', 'stf.STATIC', 'stf.ADVANCE', 'stf.DRAWN', 'None')
    if base_class_name == 'Chase':
        return ('ec.EMPTY', 'ec.TARGET', 'ec.AGENT', 'ec.WALL', 'ec.DRAWN', 'ec.LEFT_ARROW', 'ec.RIGHT_ARROW', 'ec.UP_ARROW', 'ec.DOWN_ARROW', 'None')
    if base_class_name == 'ReachForTheStar':
        return ('rfts.EMPTY', 'rfts.AGENT', 'rfts.STAR', 'rfts.DRAWN', 'rfts.LEFT_ARROW', 'rfts.RIGHT_ARROW', 'None')
    if base_class_name == 'PlayingWithXYZ':
        return ('xyz.EMPTY','xyz.PASS','xyz.X','xyz.Y','xyz.Z','xyz.START')
    if base_class_name == 'UnityGame':
        return ('unity.P','unity.P_Clicked','unity.S','unity.S_CLicked','unity.B','unity.START','unity.PASS','unity.CLICK',"None",
                'unity.CUBE_RED','unity.CUBE_RED_Clicked','unity.CUBE_BLUE','unity.CUBE_BLUE_Clicked','unity.CUBE_GREEN',
                'unity.CUBE_GREEN_Clicked','unity.CUBE_BLACK','unity.CUBE_BLACK_Clicked','unity.CUBE_BROWN','unity.CUBE_BROWN_Clicked',
                'unity.CUBE_PINK','unity.CUBE_PINK_Clicked','unity.CUBE_YELLOW','unity.CUBE_YELLOW_Clicked','unity.CUBE_GREY','unity.CUBE_GREY_Clicked')


    raise Exception("Unknown class name", base_class_name)
