from model import *
class Player:
    def __init__(self,env,load_path):
        self.env=env
        model = Model(num_action=7)  # these values are not needed. fix dependecy later.
        self.model.load_weights(load_path)  # check this put
        self.current_observation=self.env.reset()

    def play(self):
        for i in range(0,300):
            predicted_action= self.model.step(self.current_observation)
            self.current_observation,rew,info,done=self.env.step(predicted_action)
            self.env.render()


