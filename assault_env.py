# Part taken from adborghi fantastic implementation
# https://github.com/aborghi/retro_contest_agent/blob/master/fastlearner/ppo2ttifrutti_sonic_env.py
import numpy as np
import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from baselines.common.atari_wrappers import FrameStack
import flag
from collections import deque


from baselines.common.distributions import make_pdtype


# import gym_remote.client as grc


# This will be useful for stacking frames
# from baselines.common.atari_wrappers import FrameStack

# Library used to modify frames (former times we used matplotlib)
import cv2

# setUseOpenCL = False means that we will not use GPU (disable OpenCL acceleration)
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plot



class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
        - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)
        self.frame_deque = deque([np.zeros((96, 96)), np.zeros((96, 96)), np.zeros((96, 96)), np.zeros((96, 96))],
                                 maxlen=4)

    def observation(self, frame):
        # Set frame to gray

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x print(frame)96x1
        #frame = frame[35:187 , 10:,None]
        frame = frame[45:225, :, None]

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        frame = np.squeeze(frame, axis=2)
        frame=frame/255


        # if flag.DEBUG:
        #cv2.imshow("frame",frame)
        #cv2.waitKey(0)
        return self.stack_frames(frame)

    def stack_frames(self, new_frame):
        self.frame_deque.append(new_frame)
        return np.stack(self.frame_deque)

class ActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["RIGHT", "A", "B", "NOOP"]
        actions = [['NOOP'], ['RIGHT'], ['RIGHT', 'A'], ['RIGHT', 'B'], ['RIGHT', 'A', 'B']]
        #right B=run faster
        #right A B=jump faster
        self.actions = []

        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True

            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            arr = np.array([False] * 5)
            for button in action:
                arr[buttons.index(button)] = True
            self.actions.append(arr)

        print("action is ", self.actions)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, a): # pylint: disable=W0221
        print(self.actions[a])
        return self.actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):

        return reward/10


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0
        self.reward_q = collections.deque()


    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        y_pos=info['y_pos']
        # print("rew",rew)
        # self._cur_x += rew
        # rew = max(0, self._cur_x - self._max_x)
        # self._max_x = max(self._max_x, self._cur_x)
        # # print("real reward",rew)
        if flag.STUCK_REWARD:
          stuck_flag =True
          if (len(self.reward_q) == 10):
              self.reward_q.popleft()
          self.reward_q.append(rew)

          for i in list(self.reward_q):

              if i > -0.01:
                  stuck_flag = False
          if stuck_flag:
              rew = rew - 1.01
        #print(rew)
        if flag.Y_POS_REWARD:
            rew+=y_pos
            rew=rew-79
        rew=(rew) /15



        return obs,rew, done, info


def make_env(env_idx):
    """
    Create an environment with some standard wrappers.
    """


    # Make the environment


    levelList = ['SuperMarioBros-1-1-v2','SuperMarioBros-2-1-v0','SuperMarioBros-3-1-v0','SuperMarioBros-4-1-v0','SuperMarioBros-5-1-v0','SuperMarioBros-6-1-v0','SuperMarioBros-7-1-v0','SuperMarioBros-8-1-v0']


    # record_path = "./records/" + dictsMsPacman[env_idx]['state']
    #env = gym_super_mario_bros.make(levelList[env_idx])
    env = gym.make("Assault-v0")

    #SuperMarioBros-v0
    #SuperMarioBrosRandomStages
   # env = BinarySpaceToDiscreteSpaceEnv(env,RIGHT_ONLY)

    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)


    # Stack 4 frames
    # env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
  #  env = AllowBacktracking(env)
    return env


def make_train_0():
    new_env=make_env(0)
    # new_env = gym.wrappers.Monitor(new_env, "recording0")
    return new_env


def make_train_1():
    new_env = make_env(1)
    #new_env = gym.wrappers.Monitor(new_env, "recording1")
    return new_env


def make_train_2():
    new_env = make_env(2)
    #new_env = gym.wrappers.Monitor(new_env, "recording2")
    return new_env


def make_train_3():
    new_env = make_env(3)
    #new_env = gym.wrappers.Monitor(new_env, "recording3")
    return new_env


def make_train_4():
    new_env = make_env(4)
    #new_env = gym.wrappers.Monitor(new_env, "recording4")
    return new_env


def make_train_5():
    new_env = make_env(5)
    #new_env = gym.wrappers.Monitor(new_env, "recording5")
    return new_env


def make_train_6():
    new_env = make_env(6)
    #new_env = gym.wrappers.Monitor(new_env, "recording6")
    return new_env


def make_train_7():
    new_env = make_env(7)
    #new_env = gym.wrappers.Monitor(new_env, "recording7")
    return new_env


def make_train_8():
    new_env = make_env(8)
    # new_env = gym.wrappers.Monitor(new_env, "recording8")
    return new_env


def make_train_9():
    return make_env(9)


def make_train_10():
    return make_env(10)


def make_train_11():
    return make_env(11)


def make_train_12():
    return make_env(12)


def make_test_level_Green():
    return make_test()


def make_test():
    """
    Create an environment with some standard wrappers.
    """

    # Make the environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, RIGHT_ONLY)
    print(env.action_space)
    # Build the actions array
    # env = ActionsDiscretizer(env)

    # Scale the rewards
    # env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env, 6)


    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    # env = AllowBacktracking(env)
    return env
import time




#
#print(new_env.action_space)
#exit()
#
# # episode_reward=0
# while True:
#     print(new_env.action_space.sample())
#     a,b,c,d=new_env.step(new_env.action_space.sample())
#     print("reward is", b)
# #     episode_reward += b
#     new_env.render()
# #     print(episode_reward)
#     time.sleep(0.05)

#
# while True:
#    frame=[]s
#    a,b,c,d=new_env.step(6)
#    print("reward is",b)
#    # new_env.step(3)
#    # new_env.step(1)
#    # new_env.step(5)
#    # a,b,c,d=new_env.step(6)
#    # print(b)
#    new_env.render()
# #
#    time.sleep(0.05)
# action_space=new_env.action_space
#
# pdtype = make_pdtype(action_space)
#
#
# while True:
#    a0= pdtype.sample()
#
#    a,b,c,d=new_env.step(a0)
#    print("action",a0)
#    # new_env.step(3)
#    # new_env.step(1)
#    # new_env.step(5)
#    # a,b,c,d=new_env.step(6)
#    # print(b)
#    new_env.render()
#
#    time.sleep(0.05)






