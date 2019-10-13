import numpy
import random
from model import Model
import tensorflow as tf
import numpy as np
import flag
import datetime


class Runner():
    def __init__(self,num_steps,env,discount_factor):
        self.num_steps=num_steps
        self.env=env
        self.current_observation = self.env.reset()
        self.discount_factor=discount_factor

    def run(self,model):
        rewards = []
        observations = []
        dones = []
        actions=[]
        values=[]
        for j in range(self.num_steps):

            predicted_action, value = model.step(self.current_observation)
            actions.append(predicted_action[0]) #check this for multiple envs version
            values.append(value)
            observation, reward, done, info = self.env.step(predicted_action)
            if flag.SHOW_GAME:
                self.env.render()
            self.current_observation = observation
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            if done:
                self.current_observation = self.env.reset()

        advantages=self.compute_advantage(rewards, values, dones)
        return observations, rewards, actions, values, advantages, dones

    def compute_advantage(self,rewards,values,dones):
        advantages = []
        for step in range(self.num_steps):
            if dones[step] or step==(self.num_steps-1):
                advantages.append(rewards[step] - values[step])
            else:
                advantages.append(rewards[step] + self.discount_factor * values[step + 1] - values[step])
        return advantages


class Trainer():
    def __init__(self,num_training_steps,num_game_steps,num_epoch,batch_size,learning_rate,discount_factor,env,num_action,value_coef,clip_range):
        self.env=env
        self.training_steps=num_training_steps
        self.num_epoch=num_epoch
        self.batch_num=batch_size
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.num_game_steps=num_game_steps
        self.batch_size = batch_size
        self.new_model = Model(num_action)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.negative_log_p_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.old_negative_log_p = 0# make this right for the first run
        self.clip_range=clip_range
        self.value_coef=value_coef
        self.loss_avg = tf.keras.metrics.Mean()
        self.policy_loss_avg = tf.keras.metrics.Mean()
        self.value_loss_avg = tf.keras.metrics.Mean()
        assert self.num_game_steps % self.batch_size == 0
        self.batch_num=int(self.num_game_steps / self.batch_size)
        self.first_neg_log=True
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.save_interval=10


    def collect_experiance_and_train(self):

        train_runner=Runner(num_steps=self.num_game_steps,env=self.env,discount_factor=self.discount_factor)
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

            experiance = list(zip(observations,rewards,actions,values,advantages,dones))
            random.shuffle(experiance)
            for epoch in range(0,self.num_epoch):
                for n in range(0,self.batch_num):
                    start_index=n*self.batch_size
                    experiance_slice=experiance[start_index:start_index+self.batch_size]
                    observations, rewards, actions,values,advantages, dones = zip(*experiance_slice)
                    loss,policy_loss,value_loss=self.train_model(observations,rewards,actions,values,advantages,dones)
                    self.loss_avg(loss)
                    self.policy_loss_avg(policy_loss)
                    self.value_loss_avg(value_loss)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss_avg', self.loss_avg.result(), step=epoch)
                    tf.summary.scalar('policy_loss_avg', data=self.policy_loss_avg.result(), step=epoch)
                    tf.summary.scalar('value_loss_avg', data=self.value_loss_avg.result(), step=epoch)
                   # tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
                    # add more scalars

                self.loss_avg.reset_states()
            if train_step % self.save_interval==0:
                self.new_model.save_weights('./models/train'+self.current_time+'/'+'train')



    def train_model(self,observations,rewards,actions,values,advantages,dones):

            observations_array = np.array(observations)
            rewards_array = np.array(rewards)
            actions_array = np.array(actions)
            advantages_array=np.array(advantages)

            if flag.DEBUG:
                print("input observations shape", observations_array.shape)
                print("input rewards shape", rewards_array.shape)
                print("input actions shape", actions_array.shape)
                print("input advantages shape", advantages_array.shape)
                print("rewards",rewards)
                print("advantages",advantages)
                print("actions",actions)
            loss,policy_loss,value_loss,grads=self.grad(observations_array, actions_array, rewards_array, advantages_array)
            self.loss_avg(loss)
            self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_variables))
            return loss,policy_loss,value_loss


    def compute_loss(self, input_observations, rewards, actions, advantages):
        policy, predicted_value = self.new_model.forward_pass(input_observations)
        if flag.DEBUG:
            print("policy",policy)
            print("predicted value",predicted_value)

        value_loss = tf.losses.mse(predicted_value,tf.cast(rewards,dtype="float32"))
        if flag.DEBUG:
            print("value_loss",value_loss)

        negative_log_p=self.negative_log_p_object(actions,self.new_model.policy)
        if flag.DEBUG:
            print("negative_log", negative_log_p)

        if self.first_neg_log:
            ratio=tf.cast(1,dtype="float32")
        else:
            ratio = tf.exp(self.old_negative_log_p - negative_log_p)

        if flag.DEBUG:
            print("ratio", ratio)
        self.old_negative_log_p=negative_log_p

        policy_loss = -advantages * ratio
        if flag.DEBUG:
            print("policy_loss", policy_loss)

        clipped_policy_loss = -advantages * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        if flag.DEBUG:
            print("clipped_policy_loss",clipped_policy_loss)
        selected_policy_loss = tf.reduce_mean(tf.maximum(policy_loss, clipped_policy_loss))

        loss = selected_policy_loss + tf.convert_to_tensor(self.value_coef,dtype="float32") * value_loss
        if flag.DEBUG:
            print("selected_policy_loss",selected_policy_loss)
            print("LOOSSS",loss)
        return loss,policy_loss,value_loss

    def grad(self,observations, actions, rewards, advantages):
        with tf.GradientTape() as tape:

            loss,policy_loss,value_loss = self.compute_loss(observations, rewards, actions, advantages)
        return loss,policy_loss,value_loss, tape.gradient(loss, self.new_model.trainable_variables)






























