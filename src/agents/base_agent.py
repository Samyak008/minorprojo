from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def execute(self, query):
        pass

    @abstractmethod
    def respond(self):
        pass