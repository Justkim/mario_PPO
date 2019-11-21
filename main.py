from train import Trainer
from env import *
import mario_env
import gym
import flag
from play import Player
import moving_dot_env


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
    new_trainer=Trainer(num_training_steps=20000,num_env=2,num_game_steps=8,num_epoch=3,learning_rate=0.00025
                        ,discount_factor=0.99,env=env,num_action=5,clip_range=0.2,value_coef=0.5,save_interval=50,
                        log_interval=10,
                        entropy_coef=0.02,lam=0.95,mini_batch_size=8,num_action_repeat=1)
    new_trainer.collect_experiance_and_train()
elif flag.PLAY:
    env = moving_dot_env.make_train_0()
    new_player=Player(env=env,load_path='/home/kim/DeepRL_Palace/my_DeepRl_projects/mario_PPO/models/step9300-20191121-112302/train')
    new_player.play()

# else:
#     new_player=Player(env=env)
#     new_player.play(%cd PPO)

