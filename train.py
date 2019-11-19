import numpy
import random
from model import Model
import tensorflow as tf
import numpy as np
import flag
import datetime
import multiprocessing as mp
import ray
import mario_env
import moving_dot_env
import gym
from baselines import logger


@ray.remote
class Simulator(object):
    def __init__(self,num_action_repeat):
        self.env = moving_dot_env.make_train_0()
        self.env.reset()
        self.num_action_repeat=num_action_repeat

    def step(self, action):
        for i in range(self.num_action_repeat):
            observations,rewards,dones,info=self.env.step(action)
            if dones:
                observations = self.reset()
        if flag.SHOW_GAME:
            self.env.render()
        return observations, rewards, dones

    def reset(self):
        return self.env.reset()


class Trainer():
    def __init__(self,num_training_steps,num_env,num_game_steps,num_epoch,
                 learning_rate,discount_factor,env,num_action,
                 value_coef,clip_range,save_interval,log_interval,entropy_coef,lam,mini_batch_size,num_action_repeat):
        self.envs=env
        self.training_steps=num_training_steps
        self.num_epoch=num_epoch
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.num_game_steps=num_game_steps
        self.mini_batch_size = mini_batch_size
        self.num_env=num_env
        self.batch_size=num_env*num_game_steps
        self.clip_range=clip_range
        self.value_coef=value_coef
        self.entropy_coef = entropy_coef
        self.mini_batch_size=mini_batch_size
        self.num_action=num_action
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        assert self.batch_size % self.mini_batch_size == 0
        self.mini_batch_num=int(self.batch_size / self.mini_batch_size)
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + self.current_time + '/train'
        log_dir = 'logs/' + self.current_time + '/log'
        if flag.TENSORBOARD_AVALAIBLE:
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        else:
            logger.configure(dir=log_dir)
        self.save_interval=save_interval
        self.lam=lam
        self.log_interval=log_interval
        self.num_action_repeat=num_action_repeat

    if flag.ON_COLAB:
        tf.enable_eager_execution()


    def collect_experiance_and_train(self):

        self.new_model=Model(self.num_action, self.value_coef, self.entropy_coef, self.clip_range)
        if flag.LOAD:
            self.new_model.load_weights('./models/step760-20191106-153400/train') #check this put
            print("loaded model weigths from checkpoint")

        ray.init()
        current_observations = []
        runners = []
        returned_observations = []

        for i in range(self.num_env):
            runners.append(Simulator.remote(self.num_action_repeat))
            returned_observations.append(runners[i].reset.remote())
        for i in range(self.num_env):
            current_observations.append(ray.get(returned_observations[i]))



        for train_step in range(self.training_steps):

            self.loss_avg = tf.keras.metrics.Mean()
            self.policy_loss_avg = tf.keras.metrics.Mean()
            self.value_loss_avg = tf.keras.metrics.Mean()
            self.avg_entropy = tf.keras.metrics.Mean()
            returned_objects=[]
            observations=[]
            rewards=[]
            dones=[]
            values=[]
            actions=[]
            experiences = []
            total_experiances=[]
            for game_step in range(self.num_game_steps):

                observations.extend(current_observations)
                current_observations=np.array(current_observations)
                decided_actions, predicted_values = self.new_model.step(current_observations)
                values.append(predicted_values)
                actions.extend(decided_actions)
                experiences=[]
                for i in range(self.num_env):
                        returned_objects.append(runners[i].step.remote(decided_actions[i]))
                        experiences.append(ray.get(returned_objects[i]))
                current_observations=[each[0] for each in experiences]
                rewards.append([each[1] for each in experiences])
                dones.append([each[2] for each in experiences])

            decided_actions, predicted_values = self.new_model.step(current_observations)
            values.append(predicted_values)

            observations_array=np.array(observations)


            rewards_array = np.array(rewards)
            dones_array = np.array(dones)

            values_array=np.array(values)

            actions_array = np.array(actions)
            advantages_array,returns_array=self.compute_advantage(rewards_array,values_array,dones_array)
            values_array = values_array[:-1,:]

            # print(values_array.shape)
            # print("input observations shape", observations_array.shape)
            # print("input rewards shape", rewards_array.shape)
            # print("input actions shape", actions_array.shape)
            # print("input advantages shape", advantages_array.shape)
            # print("values shape", values_array.shape)
            values_array=values_array.flatten()
            # returns_array = returns_array.flatten()

            # print("input observations shape", observations_array.shape)
            # print("input rewards shape", rewards_array.shape)
            # print("input actions shape", actions_array.shape)
            # print("input advantages shape", advantages_array.shape)
            # print("values shape", values_array.shape)


            # actions_array=np.swapaxes(actions_array,0,1)
            random_indexes=np.arange(self.batch_size)
            np.random.shuffle(random_indexes)


            for epoch in range(0,self.num_epoch):
                for n in range(0,self.mini_batch_num):
                    start_index=n*self.mini_batch_size
                    index_slice=random_indexes[start_index:start_index+self.mini_batch_size]
                    experience_slice=(arr[index_slice] for arr in (observations_array,returns_array,actions_array,
                                                                   values_array,advantages_array))
                    loss, policy_loss, value_loss, entropy=self.train_model(*experience_slice)
                    self.loss_avg(loss)
                    self.policy_loss_avg(policy_loss)
                    self.value_loss_avg(value_loss)
                    self.avg_entropy(entropy)


            loss_avg_result=self.loss_avg.result()
            policy_loss_avg_result=self.policy_loss_avg.result()
            value_loss_avg_result=self.value_loss_avg.result()
            entropy_avg_result=self.avg_entropy.result()
            print("training step {:03d}, Epoch {:03d}: Loss: {:.3f}, policy loss: {:.3f}, value loss: {:.3f}, entopy: {:.3f} ".format(train_step,epoch,
                                                                         loss_avg_result,
                                                                        policy_loss_avg_result,
                                                                         value_loss_avg_result,
                                                                         entropy_avg_result))
            if flag.DEBUG:
                print("policy", self.new_model.probs)
            if flag.TENSORBOARD_AVALAIBLE:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss_avg', loss_avg_result, step=train_step)
                    tf.summary.scalar('policy_loss_avg', data=policy_loss_avg_result, step=train_step)
                    tf.summary.scalar('value_loss_avg', data= value_loss_avg_result, step=train_step)
                    tf.summary.scalar('entropy_avg', data=entropy_avg_result, step=train_step)
                    tf.summary.scalar('rewards_avg', data=np.average(rewards), step=train_step)
                   # tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
                # add more scalars
            else:
                if train_step % self.log_interval == 0:
                    logger.record_tabular("train_step", train_step)
                    logger.record_tabular("loss", loss_avg_result.numpy())
                    logger.record_tabular("value loss",  value_loss_avg_result.numpy())
                    logger.record_tabular("policy loss", policy_loss_avg_result.numpy())
                    logger.record_tabular("entropy", entropy_avg_result.numpy())
                   # logger.record_tabular("policy", self.new_model.dist.numpy())
                    logger.dump_tabular()

            self.loss_avg.reset_states()
            self.policy_loss_avg.reset_states()
            self.value_loss_avg.reset_states()
            self.avg_entropy.reset_states()

            if train_step % self.save_interval==0:
                self.new_model.save_weights('./models/step'+str(train_step)+'-'+self.current_time+'/'+'train')

    def compute_advantage(self, rewards, values, dones):
        # print("rewards are",rewards)
        advantages = []
        last_advantage = 0
        for step in reversed(range(self.num_game_steps)):
            is_there_a_next_state = 1.0 - dones[step]
            delta = rewards[step] + is_there_a_next_state * self.discount_factor * values[step + 1] - values[step]
            if flag.USE_GAE:
                    advantage = last_advantage = delta + self.discount_factor * \
                                                 self.lam * is_there_a_next_state * last_advantage
                    advantages.extend(advantage)
            else:
                    advantages.append(delta)
        if flag.USE_GAE:
            advantages.reverse()
        advantages=np.array(advantages)
        returns=advantages+values.flatten()[-1:]
        return advantages,returns


    def train_model(self,observations_array,rewards_array,actions_array,values_array,advantages_array):

            if flag.USE_STANDARD_ADV:
                advantages_array=advantages_array.mean() / (advantages_array.std() + 1e-13)

            if flag.DEBUG:
                print("input observations shape", observations_array.shape)
                print("input rewards shape", rewards_array.shape)
                print("input actions shape", actions_array.shape)
                print("input advantages shape", advantages_array.shape)
                print("values shape",values_array.shape)

                print("rewards",rewards_array)
                print("advantages",advantages_array)
                print("actions",actions_array)
                print("values",values_array)
            loss,policy_loss,value_loss,entropy,grads=self.new_model.grad(observations_array, actions_array, rewards_array, values_array,advantages_array)
            self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_variables))
            return loss,policy_loss,value_loss,entropy































