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
import gym
from baselines import logger


@ray.remote
class Simulator(object):
    def __init__(self,num_action_repeat):
        self.env = mario_env.make_train_0()
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




class Runner():
    def __init__(self,num_steps,env,discount_factor,lam,model,queue):
        self.q = queue
        self.num_steps=num_steps
        self.discount_factor=discount_factor
        self.lam=lam
        self.env=env
        self.total_steps=0
        self.current_observation = env.reset()
        self.model=model
        self.remote_env=Simulator.remote()



    def run(self):

        import tensorflow as tf
        rewards = []
        observations = []
        dones = []
        actions=[]
        values=[]
        max_step_exceed=False
        print("hehe")
        for j in range(self.num_steps):
            print("lay")
            observations.append(self.current_observation)
            # self.lock.acquire()
            #new_model = Model(7,1,1,1)
            predicted_action, value = self.model.step(self.current_observation)

            # self.lock.release()

            print("lala")
            actions.append(predicted_action[0]) #check this for multiple envs version
            values.append(value[0])
            if flag.MARIO_ENV:
                print("1")
                returned_object=self.remote_env.step.remote(predicted_action[0])
                observation, reward, done, info = ray.get(returned_object)
                print("2")
            else:
                observation, reward, done, info = self.env.step(predicted_action)
            if flag.SHOW_GAME:
                self.env.render()
            self.current_observation = observation
            observations.append(self.current_observation)
            rewards.append(reward)
            dones.append(done)
            # self.total_steps+=1
            # if self.total_steps>=self.max_steps:
            #     max_step_exceed=True

            if done:
                self.current_observation = self.env.reset()
                print("Done")


        advantages=self.compute_advantage(rewards, values, dones)
        # self.q.put((observations, rewards, actions, values, advantages, dones))
        self.q.put((1))
        self.q.close()

        #return observations, rewards, actions, values, advantages, dones

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



    def collect_experiance_and_train(self):

        self.new_model=Model(self.num_action, self.value_coef, self.entropy_coef, self.clip_range)
        if flag.LOAD:
            self.new_model.load_weights('./models/step220-20191103-113750/train') #check this put
            print("loaded model weigths from checkpoint")

        ray.init()
        current_observations = []
        runners = []
        returned_observations = []

        for i in range(self.num_env):
            #new_simulator=Simulator()
            runners.append(Simulator.remote(self.num_action_repeat))
            returned_observations.append(runners[i].reset.remote())
        for i in range(self.num_env):
            current_observations.append(ray.get(returned_observations[i]))
        current_observations_array=np.array(current_observations)


        for train_step in range(self.training_steps):

            self.loss_avg = tf.keras.metrics.Mean()
            self.policy_loss_avg = tf.keras.metrics.Mean()
            self.value_loss_avg = tf.keras.metrics.Mean()
            self.avg_entropy = tf.keras.metrics.Mean()
            experiences=[]
            returned_objects=[]
            #self.new_model.step(np.array(current_observations)
            observations=[]
            rewards=[]
            dones=[]
            # for i in range(self.num_env):
            #     observations.append([])
            #     rewards.append([])
            #     dones.append([])
            current_observations_list=[]
            values=[]
            actions=[]
            # done_flags=[False for x in range(0,self.num_env)]
            observations.extend(numpy.ndarray.tolist(current_observations_array))
            for game_step in range(self.num_game_steps):
                actions_, values_ = self.new_model.step(np.array(current_observations_array))
                values.extend(np.ndarray.tolist(values_))
                actions.extend(np.ndarray.tolist(actions_))
                current_observations_list = []
                for i in range(self.num_env):
                        returned_objects.append(runners[i].step.remote(actions_[i]))
                        experiences=ray.get(returned_objects[i])
                        rewards.append(experiences[1])
                        dones.append(experiences[2])
                        observations.append(experiences[0])
                        current_observations_list.append(experiences[0])
                current_observations_array=np.array(current_observations_list)

            # observations_array = np.array([each[0] for each in experiences], ndmin=3)
            # rewards_array = np.array([each[1] for each in experiences])
            # dones_array = np.array([each[2] for each in experiences])
            # observations_array=np.array(observations)
            # rewards_array=np.array(rewards)
            # dones_array=np.array(dones)

        #    exit()

            advantages=self.compute_advantage(rewards,values,dones)

            experience=list(zip(observations,rewards,actions,values,advantages))
            for epoch in range(0,self.num_epoch):

                for n in range(0,self.mini_batch_num):
                    start_index=n*self.mini_batch_size
                    experiance_slice=experience[start_index:start_index+self.batch_size]
                    observations, rewards, actions,values,advantages = zip(*experiance_slice)

                    loss, policy_loss, value_loss, entropy=self.train_model(observations,rewards,actions,values,advantages)
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
        advantages = []
        last_advantage = 0
        for env in range(self.num_env):
            for step in reversed(range(self.num_game_steps)):
                total_step_index=env*self.num_game_steps+step
                if dones[total_step_index] or step == (self.num_game_steps - 1):
                    advantages.append(rewards[ total_step_index] - values[ total_step_index])
                else:
                    if flag.USE_GAE:
                        delta = rewards[ total_step_index] + self.discount_factor * values[ total_step_index + 1] - values[ total_step_index]
                        advantage = last_advantage = delta + self.discount_factor * self.lam * last_advantage
                        advantages.append(advantage)
                    else:
                        advantages.append(rewards[ total_step_index] + self.discount_factor * values[ total_step_index + 1] - values[ total_step_index])
            if flag.USE_GAE:
                advantages.reverse()

        return advantages


    def train_model(self,observations,rewards,actions,values,advantages):
            #print("observations shape",len(observations))
            observations_array = np.array(observations)
            rewards_array = np.array(rewards)
            actions_array = np.array(actions)
            advantages_array=np.array(advantages)
            values_array=np.array(values)
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































