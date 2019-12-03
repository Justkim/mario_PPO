from train import Trainer
import flag
from play import Player
import moving_dot_env


if flag.TRAIN:
    new_trainer = Trainer(num_training_steps=20000, num_env=1, num_game_steps=16, num_epoch=4, learning_rate=1e-4
                          , discount_factor=0.99, num_action=5, clip_range=0.2, value_coef=0.5,
                          save_interval=20,
                          log_interval=10,
                          entropy_coef=0.05, lam=0.95, mini_batch_size=16, num_action_repeat=1)
    new_trainer.collect_experiance_and_train()
elif flag.PLAY:
    env = moving_dot_env.make_train_0()
    new_player=Player(env=env,load_path='/home/kim/DeepRL_Palace/my_DeepRl_projects/mario_PPO/trains/17/step11820-20191130-072433/train')
    new_player.play()

# else:
#     new_player=Player(env=env)
#     new_player.play(%cd PPO)

