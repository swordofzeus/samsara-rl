from abc import ABC, abstractmethod

class Search(ABC):
    '''
        Select next action given policy, state, q table
    '''

    def __init__(self):
        pass

    @abstractmethod
    def step(self, policy, state, q_table):
        pass