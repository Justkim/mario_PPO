import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flag
class Model(tf.keras.Model):
    def __init__(self,num_action):
        super(Model,self).__init__(name='')
        #tf.keras.backend.set_floatx('float64')
        self.num_action=num_action
        if flag.MARIO_ENV:
            channels_order = "channels_last"
        else:
            channels_order = "channels_first"
        #add batch normilization later
        self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),data_format=channels_order,name="conv1")
        self.batch_norm1=tf.keras.layers.BatchNormalization(trainable=True, epsilon = 1e-5,name="batchnorm1")
        self.activ1=tf.keras.layers.Activation('elu')
        self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),data_format=channels_order,name="conv2")
        self.activ2 = tf.keras.layers.Activation('elu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-5, name="batchnorm2")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),data_format=channels_order,name="conv3")
        self.batch_norm3 = tf.keras.layers.BatchNormalization(trainable=True, epsilon=1e-5, name="batchnorm3")
        self.activ3 = tf.keras.layers.Activation('elu')
        self.flatten=tf.keras.layers.Flatten(name="flatten")
        self.fc1=tf.keras.layers.Dense(512,activation='elu',name="fc1")
        self.policy_layer=tf.keras.layers.Dense(self.num_action,activation='elu',name="policy_tensor", kernel_regularizer=tf.keras.initializers.VarianceScaling) #maybe use variance_scaling_initializer?


    def forward_pass(self,input_observations):
       # print(input_observations.shape)
        input_observations = tf.cast(input_observations, tf.float32)
        x=self.conv1(input_observations)
        x=self.batch_norm1(x)
        x=self.activ1(x)
        #print(x.shape)
        x=self.conv2(x)
        x=self.batch_norm2(x)
        x=self.activ2(x)
        #print(x.shape)
        x=self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activ3(x)
        #print(x.shape)
        x=self.flatten(x)
        #print(x.shape)
        x=self.fc1(x)
        #print(x.shape)
        self.policy=self.policy_layer(x)
        return self.policy

        # self.q = tf.reduce_mean(tf.multiply(self.policy_layer,one_hot_actions))


    def step(self,input_observations):
        self.forward_pass(np.expand_dims(input_observations, 0))
        return np.argmax(self.policy)

    def compute_loss(self, input_observations, actions, target_qs):
        policy=self.forward_pass(input_observations)
        one_hot_actions=self.make_one_hot(actions,self.num_action)
        self.q = tf.reduce_sum(tf.multiply(policy, one_hot_actions),axis=1)
        print("self q",self.q)
        print("target q",target_qs)
        loss=tf.keras.losses.mse(target_qs,self.q)
        return loss

    def make_one_hot(self,actions,num_actions):
        res = np.eye(num_actions)[np.array(actions).reshape(-1)]
        return res.reshape(list(actions.shape) + [num_actions])

    def grad(self,observations, actions,target_qs):
        with tf.GradientTape() as tape:
            loss= self.compute_loss(observations, actions, target_qs)
        return loss, tape.gradient(loss, self.trainable_variables)












