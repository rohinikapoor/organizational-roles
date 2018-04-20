import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import constants
import utils

from model import Model


class Model3(nn.Module, Model):
    """
    Contains the code for Model3.
    Architecture - sr_embedding -> linear_layer -> relu_activation -> linear_layer -> predicted_email_emb -> loss
     - One sr_embedding is calculated for each email as a concatenation of sender_emb and average of all receiver_
     emb present in the mail
     - Loss is calculated as L2 loss between predicted_email_embedding and email_representation. The email
     representation is obtained by averaging embeddings from pre-trained word2vec model
    """

    def __init__(self, pre_trained=False, load_from=None):
        # keeps track of how many times the model has seen each email_id, either as a sender or receiver
        self.emailid_train_freq = {}
        super(Model3, self).__init__()
        # embedding lookup for 150 users each have constants.USER_EMB_SIZE dimension representation
        self.embedding_layer = nn.Embedding(150, constants.USER_EMB_SIZE)
        # first hidden layer, linear layer with weights 2*constants.USER_EMB_SIZEx500
        # this should be the size of <sender+receiver representation>
        self.h1_layer = nn.Linear(2*constants.USER_EMB_SIZE, 500)
        # ReLU activation used
        self.relu = nn.ReLU()
        # final linear layer that outputs the predicted email representation
        self.email_layer = nn.Linear(500, constants.EMAIL_EMB_SIZE)
        if pre_trained:
            self.load(load_from)

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

    def train(self, emails, w2v, epochs=10, save_model=True):
        loss_criteria = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, momentum=0.0)
        email_reps = w2v.get_email_reps(emails, average=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            start = time.time()
            for i in range(len(emails)):
                sender_id = utils.get_userid(emails[i, constants.SENDER_EMAIL])
                # if the sender was not found or no representation was found for any words of the emails, ignore
                if sender_id is None or type(email_reps[i]) == type(None):
                    continue

                # gets the average email embedding based on word embeddings of all the words in the mail
                email_rep = email_reps[i]
                recv_list = emails[i, constants.RECEIVER_EMAILS].split('|')
                recv_ids = []
                for recv in recv_list:
                    recv_id = utils.get_userid(recv)
                    if recv_id is not None:
                        recv_ids.append(recv_id)
                        # if the receiver was found update the frequency count
                        self.emailid_train_freq[recv] = self.emailid_train_freq.get(recv, 0) + 1

                # if none of the receivers were found, ignore this case
                if len(recv_ids) == 0:
                    continue

                # if the sender was found and is being used for training update his freq count
                self.emailid_train_freq[emails[i, constants.SENDER_EMAIL]] = self.emailid_train_freq.get(
                    emails[i, constants.SENDER_EMAIL], 0) + 1

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
            end = time.time()
            print 'time taken for epoch : ', (end-start)
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)

        if save_model:
            file_name = constants.RUN_ID + '_model.pth'
            self.save(file_name)
        email_ids, embs = self.extract_user_embeddings()
        utils.save_user_embeddings(email_ids, embs)
        # utils.get_similar_users(email_ids, embs)
        utils.plot_with_tsne(email_ids, embs, display_hover=False)

    def get_repr(self, identifier):
        pass
