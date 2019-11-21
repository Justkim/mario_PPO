import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flag


def make_conv_layer(filters,kernel_size,channels_order,name,stride_size=1):
    # return tf.keras.layers.Conv2D(filters, kernel_size , strides=(stride_size,stride_size), activation='relu', data_format=channels_order,
    #                                         name=name,kernel_initializer=tf.keras.initializers.VarianceScaling(
    #   scale=2., mode="fan_in", distribution="uniform"))
    return tf.keras.layers.Conv2D(filters, kernel_size, strides=(stride_size, stride_size), activation=tf.keras.activations.elu,
                                  data_format=channels_order,
                                  name=name, kernel_initializer=tf.keras.initializers.glorot_normal(1.0))
def make_dense_layer(units,name,activation=None):
    # return tf.keras.layers.Dense(units=units,activation=activation,name=name,kernel_initializer=tf.keras.initializers.VarianceScaling(
    #   scale=2., mode="fan_in", distribution="uniform"))
    return tf.keras.layers.Dense(units=units, activation=activation, name=name,
                                 kernel_initializer=tf.keras.initializers.glorot_normal(1.0))



class TargetModel(tf.keras.Model):
    def __init__(self,num_action,value_coef,entropy_coef,clip_range):
        super(TargetModel,self).__init__(name='')
        tf.keras.backend.set_floatx('float64')
        self.num_action=num_action
        channels_order = "channels_first"
        # self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),activation='elu',data_format=channels_order,name="conv1")
        # self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),activation='elu',data_format=channels_order,name="conv2")
        # self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),activation='elu',data_format=channels_order,name="conv3")

        self.conv1=make_conv_layer(32, 8, channels_order, "conv1", 4)
        self.conv2=make_conv_layer(64, 4, channels_order, "conv2", 2)
        self.conv3=make_conv_layer(64, 3, channels_order, "conv3", 1)
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.fc1 = make_dense_layer(512, "fc1")

    def forward_pass(self,input_observations):
        #print(input_observations.shape)
        x=self.conv1(input_observations)
        #print(x.shape)
        x=self.conv2(x)
        x=self.conv3(x)
        #print(x.shape)
        x=self.flatten(x)
        #print(x.shape)
        self.target_value=self.fc1(x)
        #print(x.shape)
        return self.target_value


class PredictorModel(tf.keras.Model):
    def __init__(self, num_action, value_coef, entropy_coef, clip_range):
        super(TargetModel, self).__init__(name='')
        tf.keras.backend.set_floatx('float64')
        self.num_action = num_action
        channels_order = "channels_first"
        # self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),activation='elu',data_format=channels_order,name="conv1")
        # self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),activation='elu',data_format=channels_order,name="conv2")
        # self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),activation='elu',data_format=channels_order,name="conv3")

        self.conv1 = make_conv_layer(32, 8, channels_order, "conv1", 4)
        self.conv2 = make_conv_layer(64, 4, channels_order, "conv2", 2)
        self.conv3 = make_conv_layer(64, 3, channels_order, "conv3", 1)
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.fc1 = make_dense_layer(512, "fc1",activation='relu')
        self.fc2 = make_dense_layer(512, "fc1",activation='relu')
        self.fc3 = make_dense_layer(512, "fc1")

    def forward_pass(self, input_observations):
        # print(input_observations.shape)
        x = self.conv1(input_observations)
        # print(x.shape)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        self.predictor_value = self.fc3(x)
        return self.predictor_value
