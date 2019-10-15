from train import Trainer
from env import *
import mario_env
import gym
import gym_minigrid
import flag



if flag.MARIO_ENV:
    env=mario_env.make_train_0()
else:
    env = gym.make('MiniGrid-Empty-5x5-v0')
    env = gym_minigrid.wrappers.RGBImgObsWrapper(
        env)  # attention: I changed render parameters in wrappers file to make this work

    # env=env.make_train_0()
    env = processObservation(env, (86, 86))
    env = gym_minigrid.wrappers.StateBonus(env)

if flag.TRAIN:

    new_trainer=Trainer(num_training_steps=20000,num_game_steps=8,num_epoch=4,batch_size=4,learning_rate=0.00045,discount_factor=0.99,env=env,num_action=7,clip_range=0.2,value_coef=0.5,save_interval=50,entropy_coef=0.001,lam=0.95)
    new_trainer.collect_experiance_and_train()
# else:
#     new_player=Player(env=env)
#     new_player.play()

