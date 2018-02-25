import time
from model import Model
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import utils
import numpy as np


class Model3(nn.Module, Model):
    """
    Contains the code for Model3.
    Architecture - sr_embedding -> linear_layer -> relu_activation -> linear_layer -> predicted_email_emb -> loss
     - One sr_embedding is calculated for each email as a concatenation of sender_emb and average of all receiver_
     emb present in the mail
     - Loss is calculated as L2 loss between predicted_email_embedding and email_representation. The email
     representation is obtained by averaging embeddings from pre-trained word2vec model
    """

    def __init__(self, epochs=10):
        self.epochs = epochs
        super(Model3, self).__init__()
        # embedding lookup for 150 users each have 50 dimension representation
        self.embedding_layer = nn.Embedding(150, 50)
        # first hidden layer, linear layer with weights 200x500
        # this should be the size of <sender+receiver representation>
        self.h1_layer = nn.Linear(100, 500)
        # ReLU activation used
        self.relu = nn.ReLU()
        # final linear layer that outputs the predicted email representation
        self.email_layer = nn.Linear(500, 50)

    def forward(self, s_id, r_ids):
        """
        Input is an integer sender_id and a list of receiver_ids. These ids are first converted into torch Variables and
        then are used to lookup embeddings. If there are multiple receivers, an average embeddings is taken of all the
        receivers. The gradients in the receiver embeddings are distributed accordingly.
        Does a forward pass and returns the predicted email representation
        :param s_id:
        :param r_id
        :return: email representation
        """
        # convert integers to long tensors
        s_id = autograd.Variable(torch.LongTensor([s_id]))
        r_ids = autograd.Variable(torch.LongTensor(r_ids))
        # extract the embedding for all the receivers and take an average
        r_emb = torch.mean(self.embedding_layer(r_ids), 0, True)
        # extract the sender embedding
        s_emb = self.embedding_layer(s_id)
        # simple concatenation of sender and receiver embedding
        sr_emb = torch.cat((s_emb, r_emb), 1)
        h1 = self.relu(self.h1_layer(sr_emb))
        email_reps = self.email_layer(h1)
        return email_reps

    def get_average_rep(self, word_reps):
        """
        Assumes that word_reps is a numpy 2d array with every row as vector representation of word.
        Calculates the mean across all rows to get average email representation
        :param word_reps:
        :return:
        """
        return np.mean(word_reps, axis=0)

    def train(self, emails, w2v):
        loss_criteria = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.6, momentum=0.6)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(len(emails)):
                sender_id = utils.get_userid(emails[i, 0])
                email_word_reps = w2v.get_sentence(emails[i, 2])
                # if the sender was not found or no representation was found for any words of the emails, ignore
                if sender_id is None or len(email_word_reps) == 0:
                    continue
                email_rep = self.get_average_rep(email_word_reps)
                recv_list = emails[i, 1].split('|')
                recv_ids = []
                for recv in recv_list:
                    recv_id = utils.get_userid(recv)
                    if recv_id is not None:
                        recv_ids.append(recv_id)
                # if none of the receivers were found, ignore this case
                if len(recv_ids) == 0:
                    continue
                optimizer.zero_grad()
                # do the forward pass
                pred_email_rep = self.forward(sender_id, recv_ids)
                # compute the loss
                loss = loss_criteria(pred_email_rep, autograd.Variable(torch.from_numpy(email_rep)))
                # propagate the loss backward and compute the gradient
                loss.backward()
                # change weights based on gradient value
                optimizer.step()
                epoch_loss += loss.data.numpy()
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def get_repr(self, identifier):
        pass
