def run_single_episode(env, policy, record_video=False, video_out_path=None, max_num_steps=100):
    if record_video:
        env.start_recording_video(video_out_path=video_out_path)

    obs = env.reset()
    total_reward = 0.
    
    for t in range(max_num_steps):
        action = policy(obs)
        if isinstance(action,int) and action == -1:
            print("Teach me this please: \n {}".format(obs))
            return None 
        b = {0:'pass', 1:'x', 2:'y', 3:'z'}
        action = (b[action[2]],  (action[0], action[1]))

        new_obs, reward, done, debug_info = env.step(action)
        total_reward += reward

        obs = new_obs

        if done:
            break

    env.close()

    return total_reward > 0
