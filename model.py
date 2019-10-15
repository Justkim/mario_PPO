import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
class Model(tf.keras.Model):
    def __init__(self,num_action,batch_size):
        super(Model,self).__init__(name='')
        tf.keras.backend.set_floatx('float64')
        self.num_action=num_action
        self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),activation='elu',data_format="channels_first",name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),activation='elu',data_format="channels_first",name="conv2")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),activation='elu',data_format="channels_first",name="conv3")
        self.flatten=tf.keras.layers.Flatten(name="flatten")
        self.fc1=tf.keras.layers.Dense(512,activation='elu',name="fc1")
        self.value=tf.keras.layers.Dense(1,name="value_layer")
        self.policy_layer=tf.keras.layers.Dense(self.num_action,activation='elu',name="policy_tensor") #maybe use variance_scaling_initializer?
        # self.dist = tf.compat.v1.distributions.Categorical(logits=policy)
        self.softmax_layer=tf.keras.layers.Softmax(name="softmax")
        self.actions_batch=np.zeros(batch_size)


    def forward_pass(self,input_observations):
        print(input_observations.shape)
        x=self.conv1(input_observations)
        print(x.shape)
        x=self.conv2(x)
        #print(x.shape)
        x=self.conv3(x)
        #print(x.shape)
        x=self.flatten(x)
        #print(x.shape)
        x=self.fc1(x)
        print(x.shape)
        predicted_value=self.value(x)[:,0] #try sum values in one axis
        #print(predicted_value.shape)
        self.policy=self.policy_layer(x)
        #self.policy=tf.maximum(self.policy_layer(x),1e-13)
        print("IMP",self.policy)
        self.dist = tfp.distributions.Categorical(self.policy)
        action=self.dist.sample()

        #self.dist(self.policy)
        #print(self.policy.shape)
        # print(policy.shape)
        #print(self.policy)
        #print(predicted_value)
        return action,predicted_value

    def step(self,input_observations):
        observations=np.expand_dims(input_observations,0)
        #print(observations.shape)
        action,predicted_value=self.forward_pass(observations)

        return action,predicted_value









