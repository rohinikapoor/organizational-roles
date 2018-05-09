import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import constants

import utils

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
    def get_repr(self, identifier):
        """Given an identifier(?), get the sender or recipient's vector representation
        TODO: Consider refactoring based on requirements"""
        pass

    def save(self, filename):
        """
        Saves the model weights to file so that they can be later loaded for inference purposes
        """
        print 'Saving the model'
        torch.save(self.state_dict(), constants.MODEL_DIR + filename)

    def load(self, filename):
        """
        Loads the model weights from a given file
        """
        self.load_state_dict(torch.load(constants.MODEL_DIR + filename))

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

    def create_hidden_layers(self, inp_dim, hidden_dims, use_batchnorm):
        hidden_layers = nn.Sequential()
        for i, hd_dim in enumerate(hidden_dims):
            hidden_layers.add_module('linear' + str(i), nn.Linear(inp_dim, hd_dim))
            out_dim = hd_dim
            if use_batchnorm:
                hidden_layers.add_module('batchnorm' + str(i), nn.BatchNorm1d(out_dim))
            hidden_layers.add_module('relu' + str(i), nn.ReLU())
            inp_dim = out_dim
        return hidden_layers, inp_dim
