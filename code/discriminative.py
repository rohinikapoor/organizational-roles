from model import Model
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import utils
import numpy as np
import time
import constants
import random

class DiscriminativeModel(nn.Module, Model):
    
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
    def __init__(self, epochs=10):
        self.epochs = epochs
        # keeps track of how many times the model has seen each email_id, either as a sender or receiver
        self.emailid_train_freq = {}
        super(DiscriminativeModel, self).__init__()
        #50+50+50 (sender+recr+msg)
        self.linear = nn.Linear(150, 2)
        # embedding lookup for 150 users each have 50 dimension representation
        self.embedding_layer = nn.Embedding(150, 50)

    def forward(self, s_id, r_id, email_emb):
        """
        Input are 3 autgrad variables representing sender_id and receiver_id. These ids will be used to lookup the
        embeddings. Does a forward pass and returns the predicted email representation
        :param s_id:
        :param r_id
        :return: email representation
        """
        # extract the sender and receiver embedding
        s_emb = self.embedding_layer(s_id)
        r_emb = self.embedding_layer(r_id)
        email_emb = email_emb.unsqueeze(0)
        # simple concatenation of sender and receiver embedding with the email
        sr_emb = torch.cat((s_emb, r_emb,email_emb), 1)
        out = self.linear(sr_emb)
        return out

    def train(self, emails, all_emails, w2v):
        all_emails_len = len(all_emails)
        loss_criteria = nn.CrossEntropyLoss()
        pos_label = autograd.Variable(torch.LongTensor([1])) #labels for correct mails
        neg_label = autograd.Variable(torch.LongTensor([0])) #labels for incorrect mails
        optimizer = optim.RMSprop(self.parameters(), lr=0.0001, alpha=0.99, momentum=0.0)
        # optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

        for epoch in range(self.epochs):
            print 'running epoch ', epoch
            start = time.time()
            epoch_loss = 0.0
            for i in range(len(emails)):
                # if i%1000 == 0:
                #     print 'email processed =', i
                sender_id = utils.get_userid(emails[i, constants.SENDER_EMAIL])
                email_word_reps = w2v.get_sentence(emails[i, constants.EMAIL_BODY])
                # if no word_rep was found for any of the words in the emails, ignore this case
                if len(email_word_reps) == 0:
                    continue
                # gets the average email embedding based on word embeddings of all the words in the mail
                email_rep = np.mean(email_word_reps, axis=0)
                recv_list = emails[i, constants.RECEIVER_EMAILS].split('|')
                for recv in recv_list:
                    optimizer.zero_grad()
                    recv_id = utils.get_userid(recv)
                    # if sender or receiver is not an enron email id, we ignore this data point
                    if sender_id is None or recv_id is None:
                        continue
                    # if valid sender and receiver pairs have been found update their frequencies
                    self.emailid_train_freq[emails[i, constants.SENDER_EMAIL]] = self.emailid_train_freq.get(
                        emails[i, constants.SENDER_EMAIL], 0) + 1
                    self.emailid_train_freq[recv] = self.emailid_train_freq.get(recv, 0) + 1
                    # do the forward pass
                    pred_out = self.forward(autograd.Variable(torch.LongTensor([sender_id])),
                                                  autograd.Variable(torch.LongTensor([recv_id])),
                                                  autograd.Variable(torch.from_numpy(email_rep)))
                    
                    loss = loss_criteria(pred_out,pos_label)
                    # propagate the loss backward and compute the gradient
                    loss.backward()
                    # change weights based on gradient value
                    optimizer.step()

                    #now train for a negative sample
                    #get a random email from all_emails where the receiver is not recr, and the mail is not empty
                    while True:
                        msg_indx = random.randint(0,all_emails_len-1)
                        if utils.get_userid(all_emails[msg_indx, constants.RECEIVER_EMAILS]) != recv_id and len(w2v.get_sentence(all_emails[msg_indx, constants.EMAIL_BODY])) != 0:
                            break

                    neg_email_word_rep = w2v.get_sentence(all_emails[msg_indx, constants.EMAIL_BODY])
                    neg_email_rep = np.mean(neg_email_word_rep, axis=0)
                    pred_neg_out = self.forward(autograd.Variable(torch.LongTensor([sender_id])),
                                                  autograd.Variable(torch.LongTensor([recv_id])),
                                                  autograd.Variable(torch.from_numpy(neg_email_rep)))
                    loss = loss_criteria(pred_neg_out,neg_label)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.data.numpy()
            end = time.time()
            print 'time taken ', (end-start)
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)

        email_ids, embs = self.extract_user_embeddings()
        utils.save_user_embeddings(email_ids, embs)
        utils.get_similar_users(email_ids, embs)
        utils.plot_with_tsne(email_ids, embs, display_hover=False)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def get_repr(self, identifier):
        pass
