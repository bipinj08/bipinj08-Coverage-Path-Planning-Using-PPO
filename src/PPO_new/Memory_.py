import numpy as np

def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype
    else:
        return type(exp)


class ReplayMemory(object):
    """
    Replay memory class for RL
    """

    def __init__(self, size):
        self.k = 0
        self.head = -1
        self.full = False
        self.size = size
        self.memory = None

    def initialize(self, experience):
        self.memory = [np.zeros(shape=[self.size] + shape(exp), dtype=type_of(exp)) for exp in experience]

    def store(self, experience):
        if self.memory is None:
            self.initialize(experience)
        if len(experience) != len(self.memory):
            raise Exception('Experience not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e

        self.head = self.k
        self.k += 1
        if self.k >= self.size:
            print("Is the size of replay memory and now memory is full", self.k)

            self.full = True
            if self.full:
                print('Memory is full now...................................', self.k)
            self.k = 0

    def sample(self):
        return self.memory


    def get_size(self):
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):
        return self.size

    def reset(self):
        self.k = 0
        self.head = -1
        self.full = False
