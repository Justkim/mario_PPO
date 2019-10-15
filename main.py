from train import Trainer
from env import *
#import mario_env
import gym
import gym_minigrid

env=gym.make('MiniGrid-Empty-5x5-v0')
env=gym_minigrid.wrappers.RGBImgObsWrapper(env)#attention: I changed render parameters in wrappers file to make this work
#env=gym_minigrid.wrappers.StateBonus(env)
# env=env.make_train_0()
env=processObservation(env,(86,86))

new_trainer=Trainer(num_training_steps=1000,num_game_steps=4,num_epoch=2,batch_size=2,learning_rate=0.01,discount_factor=0.99,env=env,num_action=3,clip_range=0.1,value_coef=0.5,save_interval=20,entropy_coef=0.01,lam=0.95)
new_trainer.collect_experiance_and_train()

