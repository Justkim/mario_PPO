from train import Trainer
import mario_env
import flag
from play import Player


def main():
    if flag.MARIO_ENV:
        env = mario_env.make_train_0()

    if flag.TRAIN:
        print("end")
        new_trainer=Trainer(num_training_steps=200000,num_epoch=2,batch_size=16,learning_rate=0.0001
                            ,discount_factor=0.99,env=env,num_action=7,save_interval=100,log_interval=50,decay_rate=0.001,num_steps=16,memory_size=100)
        print("end")
        new_trainer.collect_experience_and_train()
        print("end")
    elif flag.PLAY:
        new_player=Player(env=env,load_path='')
        new_player.play()

if __name__ == '__main__': #this is important.why?
    main()
