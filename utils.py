from generalization_grid_games.envs.playing_with_XYZ import PlayingWithXYZ
from UnityDemo.constants import *
import numpy as np

def run_single_episode(env, policy, record_video=False, video_out_path=None, max_num_steps=100, interactive=False, base_class = None):
    res = {}
    if record_video:
        env.start_recording_video(video_out_path=video_out_path)

    obs = env.reset()
    # if interactive == True:
    #     PlayingWithXYZ.initialize_figure(len(obs[1]),len(obs[1]))


    total_reward = 0.
    print(obs)
    if base_class == "PlayingWithXYZ":
        for t in range(max_num_steps):
            action = policy(obs)
            if isinstance(action,int) and action == -1:
                print("Teach me this please: \n {}".format(obs))
                res['Unkown Observation'] = obs
                res['accuracy'] = None
                env.close()
                return res
            b = {0:'pass', 1:'x', 2:'y', 3:'z', 4:'empty'}
            action = (b[action[2]],  (action[0], action[1]))

            new_obs, reward, done, debug_info = env.step(action)
            total_reward += reward
            print(action)
            # if interactive == True:
            #     playing_with_XYZ.PlayingWithXYZ.initialize_figure()

            obs = new_obs
            print(new_obs)
            if done:
                break
    else:
        for t in range(max_num_steps):
            action = policy(obs)
            pick_state_flag = False
            for possible_picks in PICKED_UP:
                pick_state = np.argwhere(obs == possible_picks)
                if len(pick_state) >= 1:
                    if len(pick_state) > 1 or pick_state_flag == True:
                        print("ERROR: More than two objects picked at the same time")
                    selected_piece = pick_state[0] 
                    pick_state_flag = True
            if pick_state_flag == True:
                print("Place highlitghted: {}".format(selected_piece))
            else:
                print("Place highlitghted: {}".format([]))
            print("Action: {}".format(action) + " Hence {}".format(obs[action]))
            new_obs, reward, done, debug_info = env.step(action)
            total_reward += reward

            obs = new_obs
            if action == (0,0):
                break
            if done:
                break


    env.close()
    res['accuracy'] = total_reward > 0

    return res
