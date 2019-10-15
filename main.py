from train import Trainer
from env import *
import mario_env
import gym
import gym_minigrid
import flag
from play import Player



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

    new_trainer=Trainer(num_training_steps=20000,num_game_steps=128,num_epoch=3,batch_size=64,learning_rate=0.0001
                        ,discount_factor=0.99,env=env,num_action=7,clip_range=0.1,value_coef=0.5,save_interval=50,
                        entropy_coef=0.02,lam=0.99)
    new_trainer.collect_experiance_and_train()
elif flag.PLAY:
    new_player=Player(env=env,load_path='')
    new_player.play()
# else:
#     new_player=Player(env=env)
#     new_player.play(%cd PPO)

