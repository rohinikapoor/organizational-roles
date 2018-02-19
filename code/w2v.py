from abc import ABCMeta, abstractmethod


class W2V:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, emails):
        """This method takes a list of emails as input
        Each email may contain sender, recipient and body
        Either we just pass the body of the mails here or extract the body in this function

        The train function will initialize and learn a word2vec model based on this data
        At the end, it should save the model"""
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def get_word(self, word):
        """Given a word (str), return the vector representation for the same"""
        pass

    @abstractmethod
    def get_sentence(self, sentence):
        """Given an email body or a sentence (str), return a list of vector representation for each word in it"""
        pass
