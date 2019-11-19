import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flag

def make_conv_layer(filters,kernel_size,channels_order,name,stride_size=1):
    return tf.keras.layers.Conv2D(filters, kernel_size , strides=(stride_size,stride_size), activation='relu', data_format=channels_order,
                                            name=name,kernel_initializer=tf.keras.initializers.VarianceScaling(
      scale=2., mode="fan_in", distribution="uniform"))
def make_dense_layer(units,name,activation=None):
    return tf.keras.layers.Dense(units=units,activation=activation,name=name,kernel_initializer=tf.keras.initializers.VarianceScaling(
      scale=2., mode="fan_in", distribution="uniform"))

class Model(tf.keras.Model):
    def __init__(self,num_action,value_coef,entropy_coef,clip_range):
        super(Model,self).__init__(name='')
        tf.keras.backend.set_floatx('float64')
        self.num_action=num_action
        if flag.MARIO_ENV:
            channels_order = "channels_first"
        else:
            channels_order = "channels_first"
        # self.conv1=tf.keras.layers.Conv2D(32, 8, strides=(4,4),activation='elu',data_format=channels_order,name="conv1")
        # self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=(1,1),activation='elu',data_format=channels_order,name="conv2")
        # self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=(1,1),activation='elu',data_format=channels_order,name="conv3")
        self.conv1=make_conv_layer(32, 8, channels_order, "conv1", 4)
        self.conv2=make_conv_layer(64, 4, channels_order, "conv2", 1)
        self.conv3=make_conv_layer(64, 3, channels_order, "conv3", 1)
        self.flatten=tf.keras.layers.Flatten(name="flatten")
        self.fc1=make_dense_layer(512, "fc1", activation='relu')
        #self.fc1=tf.keras.layers.Dense(512,activation='relu',name="fc1")
        self.value=make_dense_layer(1,"value_layer")
        #self.value=tf.keras.layers.Dense(1,name="value_layer")
        self.policy_layer = make_dense_layer(self.num_action,name="policy_tensor")
        #self.policy_layer=tf.keras.layers.Dense(self.num_action,activation='elu',name="policy_tensor", kernel_initializer=tf.keras.initializers.VarianceScaling) #maybe use variance_scaling_initializer?
        # self.dist = tf.compat.v1.distributions.Categorical(logits=policy)
        self.softmax_layer=tf.keras.layers.Softmax(name="softmax")
        self.negative_log_p_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.old_negative_log_p = 0  # make this right for the first run
        self.old_values = 0
        self.clip_range = clip_range

        self.value_coef = tf.cast(value_coef, dtype="float64")
        self.entropy_coef = tf.cast(entropy_coef, dtype="float64")
        self.first_train=True
        self.old_policy=0


    def forward_pass(self,input_observations):
       # print(input_observations.shape)
        x=self.conv1(input_observations)
        #print(x.shape)
        x=self.conv2(x)
        #print(x.shape)
        x=self.conv3(x)
        #print(x.shape)
        x=self.flatten(x)
        #print(x.shape)
        x=self.fc1(x)
        #print(x.shape)
        self.predicted_value=self.value(x)[:,0] #try sum values in one axis
        #print(predicted_value.shape)
        self.policy=self.policy_layer(x)
        #self.policy=tf.maximum(self.policy_layer(x),1e-13)
        self.dist = tfp.distributions.Categorical(self.policy)
        self.action=self.dist.sample()
        self.probs=(self.softmax_layer(self.policy)).numpy()
        randoms= np.expand_dims(np.random.rand(self.probs.shape[0]),axis=1)
        self.action=(self.probs.cumsum(axis=1)>randoms).argmax(axis=1)
        return self.action,self.predicted_value


    def step(self,observations):
        # observations=np.expand_dims(input_observations,0)
        #print(observations.shape)
        #print(observations.shape)
        #print("first forward pass")
        action,predicted_value=self.forward_pass(observations)
        return action,predicted_value.numpy()


    def compute_loss(self, input_observations, rewards, actions, values, advantages):
        #print("second forward pass")
        self.forward_pass(input_observations)

        predicted_value=self.predicted_value #had to do this, because if I use values gradiants will dissapear
        # predicted_value=values
        # if flag.DEBUG:
        #     print("policy",policy)
        #     print("predicted value",predicted_value)

        #  clipped_vf= old_value + tf.clip_by_value(train_model.vf - old_value , -clip_range , clip_range)
        # value_loss = tf.losses.mse(predicted_value, tf.cast(rewards, dtype="float64"))
        value_loss=tf.square(predicted_value - tf.cast(rewards, dtype="float64"))
        if not self.first_train and flag.VALUE_CLIP:
            clipped_value = self.old_values + tf.clip_by_value(predicted_value - self.old_values, -self.clip_range,
                                                               self.clip_range)
            clipped_value_loss = tf.square(clipped_value - tf.cast(rewards, dtype="float64"))
            value_loss = tf.reduce_mean(tf.maximum(value_loss, clipped_value_loss))
        else:
            value_loss = tf.reduce_mean(value_loss)

        negative_log_p = self.negative_log_p_object(actions, self.policy)

        if self.first_train:
            ratio = tf.cast(1, dtype="float64")
            self.first_train=False
        else:
            old_negative_log_p = self.negative_log_p_object(actions, self.old_policy)
            # print("negative log",negative_log_p)
            # print("old_negative_log",old_negative_log_p)
            # print("POLICY",self.policy)
            ratio = tf.exp(negative_log_p - old_negative_log_p)


        # print("ratio is",ratio)
        # print("advantages are",advantages)
        self.old_values=predicted_value
        policy_loss = advantages * ratio
        clipped_policy_loss = advantages * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        selected_policy_loss = -tf.reduce_mean(tf.minimum(policy_loss, clipped_policy_loss))
        entropy = tf.reduce_mean(self.dist.entropy())
        #value_coef_tensor = tf.convert_to_tensor(self.value_coef, dtype="float64")
        loss = selected_policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)
        #loss = selected_policy_loss -(self.entropy_coef * entropy)
        # print("value_loss",value_loss)
        # print("loss",loss)
        # print("selected_policy_loss", selected_policy_loss)

        if flag.DEBUG:

            print("value_loss", value_loss)
            print("negative_log", negative_log_p)
            print("ratio", ratio)
            print("policy_loss", policy_loss)
            print("clipped_policy_loss", clipped_policy_loss)
            print("selected_policy_loss", selected_policy_loss)
            print("LOOSSS", loss)
        self.old_policy = self.policy

        return loss, selected_policy_loss, value_loss, entropy


    def grad(self,observations, actions, rewards,values, advantages):
        with tf.GradientTape() as tape:
            loss,policy_loss,value_loss,entropy = self.compute_loss(observations, rewards, actions, values,advantages)
        gradients=tape.gradient(loss, self.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -0.5, 0.5))
                     for grad in gradients]
        return loss,policy_loss,value_loss,entropy,gradients












