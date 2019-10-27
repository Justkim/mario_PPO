import numpy
import random
from model import Model
import tensorflow as tf
import numpy as np
import flag
import datetime
from collections import deque
from baselines import logger

from baselines.common.runners import AbstractEnvRunner

class Runner():

    def __init__(self,env,discount_factor,memory,decay_rate,num_steps):
        self.memory=memory
        self.env=env
        self.current_observation = self.env.reset()
        self.discount_factor=discount_factor
        self.total_steps=0
        self.possible_actions = [0, 1, 2, 3, 4, 5, 6]
        self.exploration_probability=1
        self.decay_rate=decay_rate
        self.current_observation = self.env.reset()
        self.num_steps=num_steps

    def pretrain(self,batch_size):

        for i in range(batch_size):
            random_action = random.choice(self.possible_actions)
            observation, reward, done, info=self.env.step(random_action)
            if flag.SHOW_GAME:
             self.env.render()
            self.memory.add((self.current_observation,random_action,reward,observation,done))
            if done:
                self.current_observation = self.env.reset()
            else:
                self.current_observation=observation



    def run(self,model):
        for i in range(self.num_steps):
            if np.random.rand()<self.exploration_probability:
                action = random.choice(self.possible_actions)
            else:
                action= model.step(self.current_observation)

            observation, reward, done, info = self.env.step(action)

            if flag.SHOW_GAME:
                self.env.render()


            if done:
                self.current_observation = self.env.reset()
                print("Done")


            self.memory.add((self.current_observation,action,reward,observation,done))
            self.current_observation = observation
            self.exploration_probability = self.exploration_probability - self.decay_rate


class Memory():
    def __init__(self,memory_size,batch_size):
        self.memory = deque(maxlen=memory_size)
        self.sample_size=batch_size
    def add(self,slice):
        self.memory.append(slice)
    def sample(self):
        return random.sample(self.memory,self.sample_size)



class Trainer():
    def __init__(self,num_training_steps,num_epoch,
                 batch_size,learning_rate,discount_factor,env,num_action,
                save_interval,log_interval,decay_rate,num_steps,memory_size):
        if flag.ON_COLAB:
            tf.enable_eager_execution()
        self.env=env
        self.training_steps=num_training_steps
        self.num_epoch=num_epoch
        self.batch_num=batch_size
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.batch_size = batch_size
        self.num_steps=num_steps


        self.new_model = Model(num_action)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + self.current_time + '/train'
        log_dir='logs/' + self.current_time + '/log'
        if flag.TENSORBOARD_AVALAIBLE:
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_interval=save_interval
        self.memory=Memory(memory_size,batch_size)
        self.decay_rate=decay_rate
        logger.configure(dir=log_dir)
        self.log_interval=log_interval



    def collect_experience_and_train(self):
        train_runner=Runner(env=self.env,discount_factor=self.discount_factor,memory=self.memory,decay_rate=self.decay_rate,num_steps=self.num_steps)
        if flag.LOAD:
            self.new_model.load_weights('./first_train/step800-20191015-132314/train') #check this put
            print("loaded model weigths from checkpoint")
        train_runner.pretrain(self.batch_size)

        for train_step in range(self.training_steps):
            train_runner.run(self.new_model)
            self.loss_avg = tf.keras.metrics.Mean()
            experience_slice= self.memory.sample()

            for epoch in range(0,self.num_epoch):
                loss=self.train_model(experience_slice)
                self.loss_avg(loss)

                loss_avg_result=self.loss_avg.result()
                # print("training step {:03d}, Epoch {:03d}: Loss: {:.3f} ".format(train_step,epoch,
                #                                                              loss_avg_result))
                if flag.TENSORBOARD_AVALAIBLE:
                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss_avg', loss_avg_result, step=epoch)
                    # add more scalars

            if train_step % self.log_interval == 0:
                logger.record_tabular("train_step", train_step)
                logger.record_tabular("loss",  loss_avg_result.numpy())
                logger.dump_tabular()
            self.loss_avg.reset_states()
            if train_step % self.save_interval==0:
                self.new_model.save_weights('./models/step'+str(train_step)+'-'+self.current_time+'/'+'train')

    def get_target_qs(self,rewards_array,next_observation_array,dones_array):
        target_q=[]
        for i in range(self.batch_size):
            if dones_array[i]:
                target_q.append(rewards_array[i])
            else:
                next_q=self.new_model.forward_pass(np.expand_dims(next_observation_array[i],0))
                target_q.append(rewards_array[i]+self.discount_factor*np.max(next_q))

        return target_q

    def train_model(self,slice):
        observations_array = np.array([each[0] for each in slice], ndmin=3)
        actions_array = np.array([each[1] for each in slice])
        rewards_array = np.array([each[2] for each in slice])
        next_observations_array = np.array([each[3] for each in slice], ndmin=3)
        dones_array = np.array([each[4] for each in slice])
        target_qs_list=self.get_target_qs(rewards_array,next_observations_array,dones_array)
        target_qs_array=np.array(target_qs_list)
        loss,grads=self.new_model.grad(observations_array, actions_array,target_qs_array )
        self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_variables))
        return loss































