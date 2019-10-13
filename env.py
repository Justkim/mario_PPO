import gym
import gym_minigrid
import cv2
from collections import deque
import numpy as np


class processObservation(gym.Wrapper):
    def __init__(self,env,requested_obs_shape):
        super(processObservation,self).__init__(env)
        self.requested_obs_shape=requested_obs_shape
        self.frame_deque = deque([np.zeros(requested_obs_shape), np.zeros(requested_obs_shape),np.zeros(requested_obs_shape),np.zeros(requested_obs_shape)],maxlen=4)
    def step(self,action):

        obs,rew,done,info= self.env.step(action)
        obs=self.preprocess(obs)
        obs=self.stack_frames(obs)
        return obs,rew,done,info

    def reset(self):
        obs=self.env.reset()
        obs = self.preprocess(obs)
        obs = self.stack_frames(obs)
        return obs

    def preprocess(self,obs):
        cut_row_num = (int)((obs.shape[0] - self.requested_obs_shape[0]) / 2)
        cut_col_num = (int)((obs.shape[1] - self.requested_obs_shape[1]) / 2)
        obs = obs[cut_row_num:-cut_row_num, cut_row_num:-cut_col_num]
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs= obs / 255.
        return obs


    def stack_frames(self,new_frame):
        self.frame_deque.append(new_frame)
        return np.stack(self.frame_deque)






#actions manual:
    # 0:left
    # 1:right
    # 2:forwards
# env=gym.make('MiniGrid-Empty-5x5-v0')
# env=gym_minigrid.wrappers.RGBImgObsWrapper(env)#attention: I changed render parameters in weappers file to make this work
# env=processObservation(env,(86,86))
# observation=env.reset()
# print(env.action_space)
# print(env.observation_space)
# obs=env.reset()
# print(obs.shape)
# #
# for i in range(4):
#     observation, reward, done, info=env.step(1)


















