from model import Model


class Model2(Model):
    def train(self, emails, w2v):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def get_repr(self, identifier):
        pass
import time
from model import Model
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import utils
import numpy as np


class Model2(nn.Module, Model):
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
        super(Model2, self).__init__()
        # embedding lookup for 150 users each have 50 dimension representation
        self.embedding_layer = nn.Embedding(150, 50)
        # first hidden layer, linear layer with weights 200x500
        # this should be the size of <sender+receiver representation>
        self.h1_layer = nn.Linear(200, 500)
        # ReLU activation used
        self.relu = nn.ReLU()
        # final linear layer that outputs the predicted email representation
        self.email_layer = nn.Linear(500, 50)

    def forward(self, s_id, r_ids,pv_emb, nv_emb):
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
        #previous word embedding
        p_emb = torch.autograd.Variable(torch.from_numpy(pv_emb))
        p_emb = p_emb.unsqueeze(0)
        #next word embedding
        n_emb = torch.autograd.Variable(torch.from_numpy(nv_emb))
        n_emb = n_emb.unsqueeze(0)
        #concatiante sender, receiver, prev word and next word embeddings 
        srpn_emb = torch.cat([s_emb, r_emb,p_emb,n_emb], 1)
        # srpn_emb = torch.cat((srp_emb, n_emb),1)
        h1 = self.relu(self.h1_layer(srpn_emb))
        word_reps = self.email_layer(h1)
        #the predicted word rep
        return word_reps

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
            #loop over each mail
            for i in range(len(emails)):
                sender_id = utils.get_userid(emails[i, 0])
                email_content = emails[i, 2]
                #skip if either sender or email is None
                if sender_id is None or email_content is None:
                    continue
                recv_list = emails[i, 1].split('|')
                recv_ids = []
                for recv in recv_list:
                    recv_id = utils.get_userid(recv)
                    if recv_id is not None:
                        recv_ids.append(recv_id)
                # if none of the receivers were found, ignore this case
                if len(recv_ids) == 0:
                    continue
                email_words = emails[i, 2].split()
                #loop for every word in the mail
                for j in range(1,len(email_words)-1):
                    prev_word = email_words[j-1]
                    next_word = email_words[j+1]
                    email_word_rep = w2v.get_word(email_words[j]) 

                    pv_emb = w2v.get_word(prev_word)
                    nv_emb = w2v.get_word(next_word)
                    #if the previous, current or next word embedding is None, skip for the word
                    if pv_emb is None or nv_emb is None or email_word_rep is None:
                        continue
                    optimizer.zero_grad()
                    # do the forward pass
                    pred_word_rep = self.forward(sender_id, recv_ids,pv_emb,nv_emb)
                    # compute the loss
                    loss = loss_criteria(pred_word_rep, autograd.Variable(torch.from_numpy(email_word_rep)))
                    # propagate the loss backward and compute the gradient
                    loss.backward()
                    # change weights based on gradient value
                    optimizer.step()
                    epoch_loss += loss.data.numpy()
                # if i % 1000 == 0:
                #     print 'at email ' +str(i)
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def get_repr(self, identifier):
        pass
