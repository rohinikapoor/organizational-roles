import numpy as np
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import random

import constants
import dal
import plots
import utils

from model import Model


class Model4(nn.Module, Model):
    """
    Contains defination for our discriminative model.
    We train the model to learn how a sender and reciever pair usually behave.
    We train a Logistic Regression as follows:
    for each message m sent by s1 to r1,
    we train (s1,r1,m) through a linear model with just 2 classes, with a true label of 1,
    we also find a message m' which has not been sent from s1->r1. It can be sent by s1 to some other r.
    we train(s1,r1,m') with a true label of 0.
    We use the cross entropy loss to calculate our loss.
    """

    def __init__(self, pre_trained=False, load_from=None, hidden_dims=[500]):
        # keeps track of how many times the model has seen each email_id, either as a sender or receiver
        self.emailid_train_freq = {}
        super(Model4, self).__init__()
        # 50+50+50 (sender+recr+msg)
        self.linear = nn.Linear(2 * constants.USER_EMB_SIZE + constants.EMAIL_EMB_SIZE, 2)
        # embedding lookup for 150 users each have 50 dimension representation
        self.embedding_layer = nn.Embedding(150, constants.USER_EMB_SIZE)
        # first hidden layer, linear layer with weights 2*constants.USER_EMB_SIZEx500
        # this should be the size of <sender+receiver representation>
        self.h1_layer = nn.Linear(2 * constants.USER_EMB_SIZE + constants.EMAIL_EMB_SIZE, 500)
        # ReLU activation used
        self.relu = nn.ReLU()
        # final linear layer that outputs the predicted email representation
        self.output_layer = nn.Linear(500, 2)
        if pre_trained:
            self.load(load_from)

    def forward(self, s_id, r_ids, email_rep):
        """
        Input are 3 autgrad variables representing sender_id and receiver_id. These ids will be used to lookup the
        embeddings. Does a forward pass and returns the predicted email representation
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

        email_emb = autograd.Variable(torch.from_numpy(email_rep))
        # simple concatenation of sender and receiver embedding with the email
        sr_emb = torch.cat((s_emb, r_emb, email_emb), 1)
        h1 = self.relu(self.h1_layer(sr_emb))
        out = self.output_layer(h1)
        return out

    def train(self, emails, w2v, epochs=10, save_model=True):
        optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, momentum=0.0)
        pos_label = autograd.Variable(torch.LongTensor([1]))  # labels for correct mails
        neg_label = autograd.Variable(torch.LongTensor([0]))  # labels for incorrect mails

        neg_emails = dal.get_negative_emails(emails, fraction=1.0)

        for epoch in range(epochs):
            epoch_loss = 0.0
            start = time.time()
            for i in range(len(emails)):
                optimizer.zero_grad()
                loss, valid = self.predict(emails[i, :], w2v, label=pos_label, training_mode=True)
                if valid:
                    # propagate the loss backward and compute the gradient
                    loss.backward()
                    # change weights based on gradient value
                    optimizer.step()
                    epoch_loss += loss.data.numpy()
                    optimizer.zero_grad()

                loss, valid = self.predict(neg_emails[i, :], w2v, label=neg_label, training_mode=True)
                if valid:
                    # propagate the loss backward and compute the gradient
                    loss.backward()
                    # change weights based on gradient value
                    optimizer.step()
                    epoch_loss += loss.data.numpy()
            end = time.time()
            print 'time taken for epoch : ', (end - start)
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)

        if save_model:
            file_name = constants.RUN_ID + '_model.pth'
            self.save(file_name)
        email_ids, embs = self.extract_user_embeddings()
        utils.save_user_embeddings(email_ids, embs)
        # utils.get_similar_users(email_ids, embs)
        plots.plot_with_tsne(email_ids, embs, display_hover=False)

    def predict(self, email, w2v, label=None, training_mode=False):
        loss_criteria = nn.CrossEntropyLoss()

        sender_id = utils.get_userid(email[constants.SENDER_EMAIL])
        email_content = email[constants.EMAIL_BODY]
        # skip if the sender does not have an embedding or there are no words in the email
        if sender_id is None or email_content is None:
            return 0, False

        # gets the average email embedding based on word embeddings of all the words in the mail
        email_rep = np.array(w2v.get_sentence(email[2]))
        if email_rep.shape[0]:
            email_rep = np.mean(email_rep, axis=0).reshape(1, -1)
        else:
            return 0, False

        recv_list = email[1].split('|')
        recv_ids = []
        for recv in recv_list:
            recv_id = utils.get_userid(recv)
            if recv_id is not None:
                recv_ids.append(recv_id)
                self.emailid_train_freq[recv] = self.emailid_train_freq.get(recv, 0) + 1
        # if none of the receivers were found, ignore this case
        if len(recv_ids) == 0:
            return 0, False

        # if the sender was found and is being used for training update his freq count
        self.emailid_train_freq[email[constants.SENDER_EMAIL]] = self.emailid_train_freq.get(
            email[constants.SENDER_EMAIL], 0) + 1

        # do the forward pass
        pred_out = self.forward(sender_id, recv_ids, email_rep)
        # compute the loss
        if training_mode:
            loss = loss_criteria(pred_out, label)
            return loss, True
        else:
            out_probs = nn.Softmax()(pred_out)
            return out_probs, True

    def get_repr(self, identifier):
        pass