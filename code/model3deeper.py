import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import constants
import plots
import utils
import dal
import metrics

from model import Model


class Model3Deeper(nn.Module, Model):
    """
    Contains the code for Model3.
    Architecture - sr_embedding -> linear_layer -> relu_activation -> linear_layer -> predicted_email_emb -> loss
     - One sr_embedding is calculated for each email as a concatenation of sender_emb and average of all receiver_
     emb present in the mail
     - Loss is calculated as L2 loss between predicted_email_embedding and email_representation. The email
     representation is obtained by averaging embeddings from pre-trained word2vec model
    """

    def __init__(self, pre_trained=False, load_from=None, hidden_dims=[500], use_batchnorm=False):
        self.val_after_epoch = 1
        self.best_val = 0.0
        # keeps track of how many times the model has seen each email_id, either as a sender or receiver
        self.emailid_train_freq = {}
        super(Model3Deeper, self).__init__()
        # embedding lookup for 150 users each have constants.USER_EMB_SIZE dimension representation
        self.embedding_layer = nn.Embedding(150, constants.USER_EMB_SIZE)
        # first hidden layer, linear layer with weights 2*constants.USER_EMB_SIZEx500
        # this should be the size of <sender+receiver representation>
        inp_dim = 2 * constants.USER_EMB_SIZE
        self.hidden_layers, out_dim = self.create_hidden_layers(inp_dim, hidden_dims, use_batchnorm)
        # final linear layer that outputs the predicted email representation
        self.email_layer = nn.Linear(out_dim, constants.EMAIL_EMB_SIZE)
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
        s_id = autograd.Variable(torch.LongTensor([s_id]))
        r_ids = autograd.Variable(torch.LongTensor(r_ids))
        # extract the embedding for all the receivers and take an average
        r_emb = torch.mean(self.embedding_layer(r_ids), 0, True)
        # extract the sender embedding
        s_emb = self.embedding_layer(s_id)
        # simple concatenation of sender and receiver embedding
        sr_emb = torch.cat((s_emb, r_emb), 1)
        hidden_out = self.hidden_layers(sr_emb)
        email_reps = self.email_layer(hidden_out)
        return email_reps

    def train(self, emails, val_data, w2v, epochs=10, save_model=True):
        neg_val_data = dal.get_negative_emails(val_data, fraction=1.0)
        optimizer = optim.RMSprop(self.parameters(), lr=0.001, alpha=0.99, momentum=0.0)

        for epoch in range(epochs):
            epoch_loss = 0.0
            start = time.time()
            for i in range(len(emails)):
                optimizer.zero_grad()
                loss, valid = self.predict(emails[i, :], w2v)
                if valid:
                    # propagate the loss backward and compute the gradient
                    loss.backward()
                    # change weights based on gradient value
                    optimizer.step()
                    epoch_loss += loss.data.numpy()
            end = time.time()
            print 'time taken for epoch : ', (end - start)
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)
            self.run_validation(epoch, val_data, neg_val_data, w2v)

        if save_model:
            file_name = constants.RUN_ID + '_model.pth'
            self.save(file_name)
        email_ids, embs = self.extract_user_embeddings()
        utils.save_user_embeddings(email_ids, embs)
        # utils.get_similar_users(email_ids, embs)
        plots.plot_with_tsne(email_ids, embs, display_hover=False)

    def run_validation(self, epoch, val_data, neg_val_data, w2v):
        """
        runs validation every configured number of epochs and save the model, if it gives a better validation metric
        """
        if epoch % self.val_after_epoch == 0:
            start = time.time()
            print 'Starting with validation'
            res = metrics.evaluate_metrics(self, 'Model2Deeper', w2v, val_data, neg_val_data, k=500,
                                           metrics=['hits@k'])
            hits_res = res['hits@k']
            if hits_res[0] > self.best_val:
                self.best_val = hits_res[0]
                file_name = constants.RUN_ID + '_model.pth'
                self.save(file_name)
            end = time.time()
            print 'validation time:', (end - start)

    def predict(self, email, w2v):
        loss_criteria = nn.MSELoss()

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
        pred_email_rep = self.forward(sender_id, recv_ids)
        # compute the loss
        loss = loss_criteria(pred_email_rep, autograd.Variable(torch.from_numpy(email_rep)))
        return loss, True

    def get_repr(self, identifier):
        pass
