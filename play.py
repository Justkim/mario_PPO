from model import *
class Player:
    def __init__(self,env,load_path):
        self.env=env
        model = Model(num_action=7, value_coef=0.5, entropy_coef=0.01,
                      clip_range=0.1)  # these values are not needed. fix dependecy later.
        self.model.load_weights(load_path)  # check this put
        self.current_observation=self.env.reset()

    def play(self):

        for i in range(0,10):
            predicted_action, value = self.model.step(self.current_observation)
            self.current_observation,rew,info,done=self.env.step(predicted_action)
            self.env.render()


