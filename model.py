import tensorflow as tf
import numpy as np
class Model(tf.keras.Model):
    def __init__(self,num_action):
        super(Model,self).__init__(name='')
        self.num_action=num_action
        self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),activation='elu',data_format="channels_first")
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),activation='elu',data_format="channels_first")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),activation='elu',data_format="channels_first")
        self.flatten=tf.keras.layers.Flatten()
        self.fc1=tf.keras.layers.Dense(512,activation='elu')
        self.value=tf.keras.layers.Dense(1)
        self.policy_layer=tf.keras.layers.Dense(self.num_action,activation='elu') #maybe use variance_scaling_initializer?
        self.softmax_layer=tf.keras.layers.Softmax()


    def forward_pass(self,input_observations):
        print(input_observations.shape)
        x=self.conv1(input_observations)
        print(x.shape)
        x=self.conv2(x)
        print(x.shape)
        x=self.conv3(x)
        print(x.shape)
        x=self.flatten(x)
        print(x.shape)
        x=self.fc1(x)
        print(x.shape)
        predicted_value=self.value(x)[:,0] #try sum values in one axis
        print(predicted_value.shape)
        self.policy=self.policy_layer(x)
        print(self.policy.shape)
        # print(policy.shape)
        print(self.policy)
        print(predicted_value)
        return self.policy,predicted_value

    def step(self,input_observations):
        observations=np.expand_dims(input_observations,0)
        print(observations.shape)
        policy,predicted_value=self.forward_pass(observations)
        print("VALUE_TRAIN",predicted_value)
        self.dist = tf.compat.v1.distributions.Categorical(logits=policy)
        action = self.dist.sample()
        return action,predicted_value








