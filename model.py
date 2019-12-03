import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flag

def make_conv_layer(filters,kernel_size,channels_order,name,stride_size=1):
    # return tf.keras.layers.Conv2D(filters, kernel_size , strides=(stride_size,stride_size), activation='relu', data_format=channels_order,
    #                                         name=name,kernel_initializer=tf.keras.initializers.VarianceScaling(
    #   scale=2., mode="fan_in", distribution="uniform"))
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=(stride_size, stride_size), activation='relu',
                                  data_format=channels_order,
                                  name=name, kernel_initializer=tf.keras.initializers.he_normal())
def make_dense_layer(units,name,activation=None):
    # return tf.keras.layers.Dense(units=units,activation=activation,name=name,kernel_initializer=tf.keras.initializers.VarianceScaling(
    #   scale=2., mode="fan_in", distribution="uniform"))
    return tf.keras.layers.Dense(units=units, activation=activation, name=name,
                                 kernel_initializer=tf.keras.initializers.he_normal())

class Model(tf.keras.Model):
    def __init__(self,num_action):
        super(Model,self).__init__(name='')
        tf.keras.backend.set_floatx('float64')
        self.num_action=num_action
        channels_order = "channels_first"
        # self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),activation='elu',data_format=channels_order,name="conv1")
        # self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),activation='elu',data_format=channels_order,name="conv2")
        # self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),activation='elu',data_format=channels_order,name="conv3")
        if not flag.MODEL_ARCHITECTURE=="a3c":
            self.conv1=make_conv_layer(32, 8, channels_order, "conv1", 4)
            self.conv2=make_conv_layer(64, 4, channels_order, "conv2", 2)
            self.conv3=make_conv_layer(64, 3, channels_order, "conv3", 1)
            self.flatten = tf.keras.layers.Flatten(name="flatten")
            self.fc1 = make_dense_layer(512, "fc1", activation='relu')
        else:
        # A3C PAPER:
            self.conv1 = make_conv_layer(16, 8, channels_order, "conv1", 4)
            self.conv2 = make_conv_layer(32, 4, channels_order, "conv2", 2)
            self.flatten = tf.keras.layers.Flatten(name="flatten")
            self.fc1 = make_dense_layer(256, "fc1", activation='relu')
        #
        #

        #self.fc1=tf.keras.layers.Dense(512,activation='relu',name="fc1")
        self.value=make_dense_layer(1,"value_layer")
        #self.value=tf.keras.layers.Dense(1,name="value_layer")
        self.policy_layer = make_dense_layer(self.num_action,name="policy_tensor")
        #self.policy_layer=tf.keras.layers.Dense(self.num_action,activation='elu',name="policy_tensor", kernel_initializer=tf.keras.initializers.VarianceScaling) #maybe use variance_scaling_initializer?
        # self.dist = tf.compat.v1.distributions.Categorical(logits=policy)
        self.softmax_layer=tf.keras.layers.Softmax(name="softmax")




    def forward_pass(self,input_observations):
        #print(input_observations.shape)
        x=self.conv1(input_observations)
        #print(x.shape)
        x=self.conv2(x)
        if not flag.MODEL_ARCHITECTURE=="a3c":
            x=self.conv3(x)
        #print(x.shape)
        x=self.flatten(x)
        #print(x.shape)
        x=self.fc1(x)
        #print(x.shape)
        self.predicted_value=self.value(x)[:,0] #try sum values in one axis
       # print(self.predicted_value.shape)
        #print(predicted_value.shape)
        self.policy=self.policy_layer(x)
        #print(self.policy.shape)
        #self.policy=tf.maximum(self.policy_layer(x),1e-13)
        self.dist = tfp.distributions.Categorical(self.policy)
        #self.action=self.dist.sample()
        self.probs=(self.softmax_layer(self.policy)).numpy()
        if flag.PLAY:
            print("entropy is",self.dist.entropy())
            print("probs are", self.probs)
        randoms= np.expand_dims(np.random.rand(self.probs.shape[0]),axis=1)
        self.action=(self.probs.cumsum(axis=1)>randoms).argmax(axis=1)
        return self.action,self.predicted_value


    def step(self,observations):
        # observations=np.expand_dims(input_observations,0)
        #print(observations.shape)
        #print(observations.shape)
        #print("first forward pass")
        action,predicted_value=self.forward_pass(observations)
        return action,predicted_value











