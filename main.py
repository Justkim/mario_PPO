from train import Trainer
from env import *
import mario_env
import gym
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
    new_trainer=Trainer(num_training_steps=20000,num_env=16,num_game_steps=128,num_epoch=1,learning_rate=0.0002
                        ,discount_factor=0.99,env=env,num_action=5,clip_range=0.1,value_coef=0.5,save_interval=50,
                        log_interval=10,
                        entropy_coef=0.05,lam=0.95,mini_batch_size=64,num_action_repeat=1)
    new_trainer.collect_experiance_and_train()
elif flag.PLAY:
    new_player=Player(env=env,load_path='./trains/3/step1100-20191107-120331/train')
    new_player.play()

# else:
#     new_player=Player(env=env)
#     new_player.play(%cd PPO)

