import gym
import gym_minigrid
import cv2


class processObservation(gym.Wrapper):
    def __init__(self,env):
        super(processObservation,self).__init__(env)

    def step(self,action):

        obs,rew,done,info= self.env.step(action)
        obs=obs[37:-37,37:-37]
        obs=cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
        return obs,rew,done,info

env=gym.make('MiniGrid-Empty-5x5-v0')

env=gym_minigrid.wrappers.RGBImgObsWrapper(env)#attention: I changed render parameters in weappers file to make this work
env=processObservation(env)
env.reset()
print(env.action_space)
print(env.observation_space)


while True:
    observation, reward, done, info=env.step(1)
    cv2.imshow("lala",observation)
    env.render()
    cv2.waitKey(20)
    # cv2.imshow("observation",observation)
    # cv2.waitKey(0)










