from abc import ABCMeta, abstractmethod
import utils
import torch
import numpy as np
import torch.autograd as autograd

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

    def extract_user_embeddings(self, threshold=1):
        """
        saves the user embeddings as a dictionary key: emailId, value user embeddings
        :return:
        """
        all_email_ids = utils.get_user_emails()

        email_ids = []
        embeddings = []
        for e_id in all_email_ids:
            if self.emailid_train_freq.get(e_id, 0) < threshold:
                continue
            email_ids.append(e_id)
            uid = utils.get_userid(e_id)
            emb = self.embedding_layer(autograd.Variable(torch.LongTensor([uid])))
            emb_np = emb.data.numpy().reshape(-1)
            embeddings.append(emb_np)
        return email_ids, np.array(embeddings)
