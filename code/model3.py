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
        # keeps track of how many times the model has seen each email_id, either as a sender or receiver
        self.emailid_train_freq = {}
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
        optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, momentum=0.0)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            start = time.time()
            for i in range(len(emails)):
                sender_id = utils.get_userid(emails[i, 0])
                email_word_reps = w2v.get_sentence(emails[i, 2])
                # if the sender was not found or no representation was found for any words of the emails, ignore
                if sender_id is None or len(email_word_reps) == 0:
                    continue

                # gets the average email embedding based on word embeddings of all the words in the mail
                email_rep = np.mean(email_word_reps, axis=0)
                recv_list = emails[i, 1].split('|')
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
                self.emailid_train_freq[emails[i, 0]] = self.emailid_train_freq.get(emails[i, 0], 0) + 1

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

        print 'Number of entries in the dictionary ', len(self.emailid_train_freq)
        email_ids, embs = self.extract_user_embeddings()
        utils.plot_with_tsne(email_ids, embs)

    def extract_user_embeddings(self, threshold=1):
        """
        saves the user embeddings as a dictionary key: emailId, value user embeddings
        :return:
        """
        email_ids = utils.get_user_emails()
        embeddings = []
        for e_id in email_ids:
            if self.emailid_train_freq.get(e_id, 0) < threshold:
                continue
            uid = utils.get_userid(e_id)
            emb = self.embedding_layer(autograd.Variable(torch.LongTensor([uid])))
            emb_np = emb.data.numpy().reshape(-1)
            embeddings.append(emb_np)
        return email_ids, np.array(embeddings)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def get_repr(self, identifier):
        pass
