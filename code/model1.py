from model import Model
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import utils
import numpy as np


class Model1(nn.Module, Model):
    """
    Contains the code for Model1.
    Architecture - sr_embedding -> linear_layer -> relu_activation -> linear_layer -> predicted_email_emb -> loss
     - sr_embedding is calculated by concatenation of sender and receiver embeddings
     - Loss is calculated as L2 loss between predicted_email_embedding and email_representation. The email
     representation is obtained by averaging embeddings from pre-trained word2vec model
    """

    def __init__(self, epochs=10):
        self.epochs = epochs
        super(Model1, self).__init__()
        # embedding lookup for 150 users each have 50 dimension representation
        self.embedding_layer = nn.Embedding(150, 50)
        # first hidden layer, linear layer with weights 200x500
        # this should be the size of <sender+receiver representation>
        self.h1_layer = nn.Linear(100, 500)
        # ReLU activation used
        self.relu = nn.ReLU()
        # final linear layer that outputs the predicted email representation
        self.email_layer = nn.Linear(500, 50)

    def forward(self, s_id, r_id):
        """
        Input are 2 autgrad variables representing sender_id and receiver_id. These ids will be used to lookup the
        embeddings. Does a forward pass and returns the predicted email representation
        :param s_id:
        :param r_id
        :return: email representation
        """
        # extract the sender and receiver embedding
        s_emb = self.embedding_layer(s_id)
        r_emb = self.embedding_layer(r_id)
        # simple concatenation of sender and receiver embedding
        sr_emb = torch.cat((s_emb, r_emb), 1)
        h1 = self.relu(self.h1_layer(sr_emb))
        email_reps = self.email_layer(h1)
        return email_reps

    def train(self, emails, w2v):
        loss_criteria = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.6, momentum=0.6)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(len(emails)):
                sender_id = utils.get_userid(emails[i, 0])
                email_word_reps = w2v.get_sentence(emails[i, 2])
                # if no word_rep was found for any of the words in the emails, ignore this case
                if len(email_word_reps) == 0:
                    continue
                # gets the average email embedding based on word embeddings of all the words in the mail
                email_rep = np.mean(email_word_reps, axis=0)
                recv_list = emails[i, 1].split('|')
                for recv in recv_list:
                    optimizer.zero_grad()
                    recv_id = utils.get_userid(recv)
                    # if sender or receiver is not an enron email id, we ignore this data point
                    if sender_id is None or recv_id is None:
                        continue
                    # do the forward pass
                    pred_email_rep = self.forward(autograd.Variable(torch.LongTensor([sender_id])),
                                                  autograd.Variable(torch.LongTensor([recv_id])))
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
