from src.PPO_new.Memory_ import ReplayMemory
from src.CPP.State import CPPState


class TrainerParams:
    def __init__(self):
        self.rm_size = 120
        self.load_model = ""
        self.num_steps = 1e6
        self.eval_period = 5


class Trainer:
    def __init__(self, params: TrainerParams, agent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size)
        self.agent = agent
        self.prefill_bar = None


class PPOTrainer(Trainer):

    def __init__(self, params, agent):
        super().__init__(params, agent)
    def add_experience(self, state: CPPState, action, reward, next_state, prediction):
        self.replay_memory.store((state,
                                  action,
                                  reward,
                                  next_state,
                                  prediction))

    def train_agent(self):
        if self.replay_memory.full:
            print('Training started..................................................................................')
            self.agent.train(self.replay_memory.memory)
            self.replay_memory.reset()
