import numpy
import random
from model import Model
import tensorflow as tf
import numpy as np
import flag
import datetime


class Runner():
    def __init__(self,num_steps,env,discount_factor,lam):
        self.num_steps=num_steps
        self.env=env
        self.current_observation = self.env.reset()
        self.discount_factor=discount_factor
        self.lam=lam

    def run(self,model):
        rewards = []
        observations = []
        dones = []
        actions=[]
        values=[]
        for j in range(self.num_steps):
            observations.append(self.current_observation)
            predicted_action, value = model.step(self.current_observation)

            actions.append(predicted_action[0]) #check this for multiple envs version
            values.append(value[0])
            if flag.MARIO_ENV:
                observation, reward, done, info = self.env.step(predicted_action[0])
            else:
                observation, reward, done, info = self.env.step(predicted_action)
            if flag.SHOW_GAME:
                self.env.render()
            self.current_observation = observation
            observations.append(self.current_observation)
            rewards.append(reward)
            dones.append(done)

            if done:
                self.current_observation = self.env.reset()
                print("Done")


        advantages=self.compute_advantage(rewards, values, dones)
        return observations, rewards, actions, values, advantages, dones

    def compute_advantage(self,rewards,values,dones):
        advantages = []
        last_advantage=0
        for step in reversed(range(self.num_steps)):
            if dones[step] or step==(self.num_steps-1):
                advantages.append(rewards[step] - values[step])
            else:
                if flag.USE_GAE:
                    delta=rewards[step] + self.discount_factor * values[step + 1] - values[step]
                    advantage= last_advantage = delta + self.discount_factor * self.lam * last_advantage
                    advantages.append(advantage)
                else:
                    advantages.append(rewards[step] + self.discount_factor * values[step + 1] - values[step])
        if flag.USE_GAE:
            advantages.reverse()



        return advantages


class Trainer():
    def __init__(self,num_training_steps,num_game_steps,num_epoch,
                 batch_size,learning_rate,discount_factor,env,num_action,
                 value_coef,clip_range,save_interval,entropy_coef,lam):
        self.env=env
        self.training_steps=num_training_steps
        self.num_epoch=num_epoch
        self.batch_num=batch_size
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.num_game_steps=num_game_steps
        self.batch_size = batch_size

        self.clip_range=clip_range
        self.value_coef=value_coef
        self.entropy_coef = entropy_coef

        self.new_model = Model(num_action,self.batch_size,self.value_coef,self.entropy_coef,self.clip_range)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        assert self.num_game_steps % self.batch_size == 0
        self.batch_num=int(self.num_game_steps / self.batch_size)
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_interval=save_interval

        self.lam=lam


    def collect_experiance_and_train(self):
        train_runner=Runner(num_steps=self.num_game_steps,env=self.env,discount_factor=self.discount_factor,lam=self.lam)
        if flag.LOAD:
            self.new_model.load_weights('20191013-164732') #check this put
            print("loaded model weigths from checkpoint")
        train_loss = []
        for train_step in range(self.training_steps):
            observations = []
            rewards = []
            dones = []
            actions = []
            values = []
            observations, rewards, actions, values, advantages, dones=train_runner.run(self.new_model)

            self.loss_avg = tf.keras.metrics.Mean()
            self.policy_loss_avg = tf.keras.metrics.Mean()
            self.value_loss_avg = tf.keras.metrics.Mean()
            self.avg_entropy = tf.keras.metrics.Mean()

            experiance = list(zip(observations,rewards,actions,values,advantages,dones))
            random.shuffle(experiance)
            for epoch in range(0,self.num_epoch):
                for n in range(0,self.batch_num):
                    start_index=n*self.batch_size
                    experiance_slice=experiance[start_index:start_index+self.batch_size]
                    observations, rewards, actions,values,advantages, dones = zip(*experiance_slice)
                    loss,policy_loss,value_loss,entropy=self.train_model(observations,rewards,actions,values,advantages,dones)
                    self.loss_avg(loss)
                    self.policy_loss_avg(policy_loss)
                    self.value_loss_avg(value_loss)
                    self.avg_entropy(entropy)
                loss_avg_result=self.loss_avg.result()
                policy_loss_avg_result=self.policy_loss_avg.result()
                value_loss_avg_result=self.value_loss_avg.result()
                entropy_avg_result=self.avg_entropy.result()
                print("Epoch {:03d}: Loss: {:.3f}, policy loss: {:.3f}, value loss: {:.3f}, entopy: {:.3f} ".format(epoch,
                                                                             loss_avg_result,
                                                                            policy_loss_avg_result,
                                                                             value_loss_avg_result,
                                                                             entropy_avg_result))

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss_avg', loss_avg_result, step=epoch)
                    tf.summary.scalar('policy_loss_avg', data=policy_loss_avg_result, step=epoch)
                    tf.summary.scalar('value_loss_avg', data= value_loss_avg_result, step=epoch)
                   # tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
                    # add more scalars

                self.loss_avg.reset_states()
            if train_step % self.save_interval==0:
                self.new_model.save_weights('./models/train'+self.current_time+'/'+'train')


    def train_model(self,observations,rewards,actions,values,advantages,dones):
            print("observations shape",len(observations))
            observations_array = np.array(observations)
            rewards_array = np.array(rewards)
            actions_array = np.array(actions)
            advantages_array=np.array(advantages)
            values_array=np.array(values)


            if flag.DEBUG:
                print("input observations shape", observations_array.shape)
                print("input rewards shape", rewards_array.shape)
                print("input actions shape", actions_array.shape)
                print("input advantages shape", advantages_array.shape)
                print("values shape",values_array.shape)

                print("rewards",rewards)
                print("advantages",advantages)
                print("actions",actions_array)
                print("values",values_array)
            loss,policy_loss,value_loss,entropy,grads=self.new_model.grad(observations_array, actions_array, rewards_array, values_array,advantages_array)
            self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_variables))
            return loss,policy_loss,value_loss,entropy































