from train import Trainer
from env import *

env=gym.make('MiniGrid-Empty-5x5-v0')
env=gym_minigrid.wrappers.RGBImgObsWrapper(env)#attention: I changed render parameters in wrappers file to make this work
env=processObservation(env,(86,86))
new_trainer=Trainer(num_training_steps=2,num_game_steps=16,num_epoch=4,batch_size=2,learning_rate=0.01,discount_factor=0.99,env=env,num_action=3,clip_range=0.1,value_coef=0.5)
new_trainer.collect_experiance_and_train()

