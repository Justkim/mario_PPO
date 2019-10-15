import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import flag
class Model(tf.keras.Model):
    def __init__(self,num_action,batch_size,value_coef,entropy_coef,clip_range):
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
        self.negative_log_p_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.old_negative_log_p = 0  # make this right for the first run
        self.old_values = 0

        self.clip_range = clip_range

        self.value_coef = tf.cast(value_coef, dtype="float64")
        self.entropy_coef = tf.cast(entropy_coef, dtype="float64")
        self.first_train=True


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
        return self.action,self.predicted_value



    def step(self,input_observations):
        observations=np.expand_dims(input_observations,0)
        #print(observations.shape)
        #print("first forward pass")
        action,predicted_value=self.forward_pass(observations)
        return action.numpy(),predicted_value.numpy()

    def compute_loss(self, input_observations, rewards, actions, values, advantages):
        #print("second forward pass")
        self.forward_pass(input_observations)
        predicted_value=self.predicted_value #had to do this, because if I use values gradiants will dissapear

        # if flag.DEBUG:
        #     print("policy",policy)
        #     print("predicted value",predicted_value)

        #  clipped_vf= old_value + tf.clip_by_value(train_model.vf - old_value , -clip_range , clip_range)

        # value_loss = tf.losses.mse(predicted_value, tf.cast(rewards, dtype="float64"))
        value_loss=tf.keras.losses.mse(predicted_value,  tf.cast(rewards, dtype="float64"))
        if not self.first_train:
            clipped_value = self.old_values + tf.clip_by_value(predicted_value - self.old_values, -self.clip_range,
                                                               self.clip_range)
            clipped_value_loss = tf.losses.mse(clipped_value, tf.cast(rewards, dtype="float64"))
            value_loss = tf.reduce_mean(tf.maximum(value_loss, clipped_value_loss))
        else:
            value_loss = tf.reduce_mean(value_loss)
        if flag.DEBUG:
            print("value_loss", value_loss)

        negative_log_p = self.negative_log_p_object(actions, self.policy)
        if flag.DEBUG:
            print("negative_log", negative_log_p)

        if self.first_train:
            ratio = tf.cast(1, dtype="float64")
        else:
            ratio = tf.exp(self.old_negative_log_p - negative_log_p)

        if flag.DEBUG:
            print("ratio", ratio)
        self.old_negative_log_p = negative_log_p
        self.old_values=predicted_value

        policy_loss = -advantages * ratio
        if flag.DEBUG:
            print("policy_loss", policy_loss)


        clipped_policy_loss = -advantages * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        if flag.DEBUG:
            print("clipped_policy_loss", clipped_policy_loss)
        selected_policy_loss = tf.reduce_mean(tf.maximum(policy_loss, clipped_policy_loss))
        entropy = tf.reduce_mean(self.dist.entropy())
        #value_coef_tensor = tf.convert_to_tensor(self.value_coef, dtype="float64")


        loss = selected_policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy)
       
        if flag.DEBUG:
            print("selected_policy_loss", selected_policy_loss)
            print("LOOSSS", loss)
        return loss, policy_loss, value_loss, entropy


    def grad(self,observations, actions, rewards,values, advantages):
        with tf.GradientTape() as tape:
            loss,policy_loss,value_loss,entropy = self.compute_loss(observations, rewards, actions, values,advantages)
        return loss,policy_loss,value_loss,entropy, tape.gradient(loss, self.trainable_variables)












