from model import *
class Player:
    def __init__(self,env,load_path):
        self.env=env
        self.model = Model(num_action=5)  # these values are not needed. fix dependecy later.
        self.model.load_weights(load_path)  # check this put
        self.current_observation=self.env.reset()

    def play(self):

        while True:
            predicted_action, value = self.model.step(np.expand_dims(self.current_observation,0))
            print("action choosen is",predicted_action)
            self.current_observation,rew,info,done=self.env.step(predicted_action)
            print("rewards is",rew)
            self.env.render()
            input()


