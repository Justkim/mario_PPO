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
import cv2


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
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
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

        self.negative_log_p_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.old_values = 0
        self.clip_range = clip_range

        self.value_coef = tf.cast(value_coef, dtype="float64")
        self.entropy_coef = tf.cast(entropy_coef, dtype="float64")
        self.first_train = True

    if flag.ON_COLAB:
        tf.enable_eager_execution()


    def collect_experiance_and_train(self):

        self.new_model=Model(self.num_action, self.value_coef, self.entropy_coef, self.clip_range)
        self.old_model = Model(self.num_action, self.value_coef, self.entropy_coef, self.clip_range)
        self.old_model.set_weights(self.new_model.get_weights())
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

            observations=[]
            rewards=[]
            dones=[]
            values=[]
            actions=[]
            experiences = []
            total_experiances=[]
            for game_step in range(self.num_game_steps):
                returned_objects = []
                observations.extend(current_observations)
                # original=current_observations[0,0,:,:]
                # duplicate=current_observations[0, 1, :, :]
                # difference = cv2.subtract(original, duplicate)
                # if cv2.countNonZero(difference) == 0 :
                #     print("The images are completely Equal")
                # input()
                decided_actions, predicted_values = self.new_model.step(np.array(current_observations))
                # decided_actions2, predicted_values2 = self.new_model.step(np.array(observations))
                # print(predicted_values1)
                # print(predicted_values2)
                # print("KKKKKKKKKKK")
                # print("lala",predicted_values)
                values.append(predicted_values)
                actions.extend(decided_actions)
                experiences=[]
                for i in range(self.num_env):
                        returned_objects.append(runners[i].step.remote(decided_actions[i]))
                        experiences.append(ray.get(returned_objects[i]))
                current_observations=[each[0] for each in experiences]
                current_observations=np.array(current_observations)
                rewards.append([each[1] for each in experiences])
                dones.append([each[2] for each in experiences])

            decided_actions, predicted_values = self.new_model.step(np.array(current_observations))
            values.append(predicted_values)
            observations_array=np.array(observations)
            rewards_array = np.array(rewards)
            dones_array = np.array(dones)
            values_array=np.array(values)
            actions_array = np.array(actions)

            advantages_array,returns_array=self.compute_advantage(rewards_array,values_array,dones_array)

            # values_array = values_array[:-1,:]

            # print(values_array.shape)
            # print("input observations shape", observations_array.shape)
            # print("input rewards shape", rewards_array.shape)
            # print("input actions shape", actions_array.shape)
            # print("input advantages shape", advantages_array.shape)
            # print("values shape", values_array.shape)
            # values_array=values_array.flatten(0)
            # returns_array = returns_array.flatten()

            # print("input observations shape", observations_array.shape)
            # print("input rewards shape", rewards_array.shape)
            # print("input actions shape", actions_array.shape)
            # print("input advantages shape", advantages_array.shape)
            # print("values shape", values_array.shape)
            values_array=values_array[:-1]
            values_array=values_array.flatten()
            # print("total values from steps",values_array)


            # actions_array=np.swapaxes(actions_array,0,1)
            random_indexes=np.arange(self.batch_size)
            np.random.shuffle(random_indexes)

            for epoch in range(0,self.num_epoch):
                for n in range(0,self.mini_batch_num):
                    start_index=n*self.mini_batch_size
                    index_slice=random_indexes[start_index:start_index+self.mini_batch_size]
                    experience_slice=(arr[index_slice] for arr in (observations_array,returns_array,values_array,actions_array,
                                                                   advantages_array))
                    last_weights = self.new_model.get_weights()
                    loss, policy_loss, value_loss, entropy=self.train_model(*experience_slice)
                    self.old_model.set_weights(last_weights)
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
            #input()

    def compute_advantage(self, rewards, values, dones):
        print("ATTT",values.shape)
        print(rewards.shape)
        print(dones.shape)
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
        values=values[:-1]
        returns=advantages+values.flatten(0)
        return advantages,returns


    def train_model(self,observations_array,returns_array,values_array,actions_array,advantages_array):

            if flag.USE_STANDARD_ADV:
                advantages_array=advantages_array.mean() / (advantages_array.std() + 1e-13)
            # print("values from steps",values_array)

            if flag.DEBUG:
                print("input observations shape", observations_array.shape)
                print("input rewards shape", returns_array.shape)
                print("input actions shape", actions_array.shape)
                print("input advantages shape", advantages_array.shape)

                print("rewards",returns_array)
                print("advantages",advantages_array)
                print("actions",actions_array)

            loss,policy_loss,value_loss,entropy,grads=self.grad(observations_array, returns_array,values_array,actions_array,advantages_array)
            self.optimizer.apply_gradients(zip(grads, self.new_model.trainable_variables))
            return loss,policy_loss,value_loss,entropy




    def compute_loss(self, input_observations, returns,values, actions,advantages):

        actions_,predicted_value=self.new_model.forward_pass(input_observations)

        # predicted_value=self.new_model.predicted_value #had to do this, because if I use values gradiants will dissapear
        print("values from forward pass",predicted_value)
        print("returns",returns)


        # print("----------------------------------")
        # predicted_value=values
        # if flag.DEBUG:
        #     print("policy",policy)
        #     print("predicted value",predicted_value)
        self.old_model.forward_pass(input_observations)
        old_negative_logp=self.negative_log_p_object(actions,self.old_model.policy)
        #  clipped_vf= old_value + tf.clip_by_value(train_model.vf - old_value , -clip_range , clip_range)
        # value_loss = tf.losses.mse(predicted_value, tf.cast(rewards, dtype="float64"))
        value_loss=tf.square(predicted_value - tf.cast(returns, dtype="float64"))
        if not self.first_train and flag.VALUE_CLIP:
            clipped_value = self.old_values + tf.clip_by_value(predicted_value - self.old_values, -self.clip_range,
                                                               self.clip_range)
            clipped_value_loss = tf.square(clipped_value - tf.cast(returns, dtype="float64"))
            value_loss = tf.reduce_mean(tf.maximum(value_loss, clipped_value_loss))
        else:
            value_loss = tf.reduce_mean(value_loss)

        negative_log_p = self.negative_log_p_object(actions, self.new_model.policy)

        if self.first_train:
            ratio = tf.cast(1, dtype="float64")
            self.first_train=False
        else:
            # print("negative log",negative_log_p)
            # print("old_negative_log",old_negative_log_p)
            # print("POLICY",self.policy)
            ratio = tf.exp(negative_log_p - old_negative_logp)


        # print("ratio is",ratio)
        # print("advantages are",advantages)
        self.old_values=predicted_value
        policy_loss = advantages * ratio
        clipped_policy_loss = advantages * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        selected_policy_loss = -tf.reduce_mean(tf.minimum(policy_loss, clipped_policy_loss))
        entropy = tf.reduce_mean(self.new_model.dist.entropy())
        #value_coef_tensor = tf.convert_to_tensor(self.value_coef, dtype="float64")
        loss = selected_policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)
        #loss = selected_policy_loss -(self.entropy_coef * entropy)
        # print("value_loss",value_loss)
        # print("loss",loss)
        # print("selected_policy_loss", selected_policy_loss)

        if True:

            print("value_loss", value_loss)
            print("negative_log", negative_log_p)
            print("ratio", ratio)
            print("policy_loss", policy_loss)
            print("clipped_policy_loss", clipped_policy_loss)
            print("selected_policy_loss", selected_policy_loss)
            print("LOOSSS", loss)


        return loss, selected_policy_loss, value_loss, entropy


    def grad(self,observations,returns,values,actions, advantages):
        with tf.GradientTape() as tape:
            loss,policy_loss,value_loss,entropy = self.compute_loss(observations, returns, values,actions,advantages)
        gradients=tape.gradient(loss, self.new_model.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -0.5, 0.5))
                     for grad in gradients]
        return loss,policy_loss,value_loss,entropy,gradients































