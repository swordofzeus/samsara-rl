from samsara_rl.functions.neural_networks.world_model_interpreter import WorldModelInterpreter


class CartPoleInterpreter(WorldModelInterpreter):
    def __init__(self):
        self.state_len = 4

    def next_state_logits(self, output):
        return output[:, 0 : self.state_len]

    def predicted_next_state(self, output):
        return output[:, 0 : self.state_len]

    def predicted_reward(self, output):
        return output[:, self.state_len]

    def predicted_terminal(self, output):
        return output[:, self.state_len + 1]

    def parse(self, output):
        return (
            self.next_state_logits(output),
            self.predicted_reward(output),
            self.predicted_terminal(output),
        )
