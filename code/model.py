from abc import ABCMeta, abstractmethod


class Model:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, emails, w2v):
        """This method takes a list of emails as input
        Each email should contain sender, recipient and body
        It also takes a word2vec model for its perusal

        The train function will update the model parameters
        It should print the objective function value as well"""
        pass

    @abstractmethod
    def save(self, filename):
        """Save the model and any supporting data structures required to recreate the state
        This might be very useful for partial training and retraining across runs on the cluster
        Might need to even save seeds and optimizer state"""
        pass

    @abstractmethod
    def load(self, filename):
        """Load the model and any supporting data structures to recreate the state
        This might be very useful for partial training and retraining across runs on the cluster
        Might need to even restore seeds and optimizer state"""
        pass

    @abstractmethod
    def get_repr(self, identifier):
        """Given an identifier(?), get the sender or recipient's vector representation
        TODO: Consider refactoring based on requirements"""
        pass
