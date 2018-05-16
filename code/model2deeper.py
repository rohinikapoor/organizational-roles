import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import constants
import plots
import utils
import metrics
import dal

from model import Model


class Model2Deeper(nn.Module, Model):
    """
    Contains the code for faster Model2.
    Architecture - [sr_embedding + prev_word + next_word] -> linear_layer -> relu_activation -> linear_layer ->
    predicted_middle_word -> loss
     - One sr_embedding is calculated for each email as a concatenation of sender_emb and average of all receiver_
     emb present in the mail
     - Loss is calculated as L2 loss between predicted_middle_word and actual middle word for the word2vec model
    """

    def __init__(self, pre_trained=False, load_from=None, hidden_dims=[500], use_batchnorm=False):
        self.val_after_epoch = 1
        self.best_val = 0.0
        self.emailid_train_freq = {}
        super(Model2Deeper, self).__init__()
        # embedding lookup for 150 users each have <constants.USER_EMB_SIZE> dimension representation
        self.embedding_layer = nn.Embedding(150, constants.USER_EMB_SIZE)
        # first hidden layer, linear layer with weights <I>x500 where
        # I = 2*constants.USER_EMB_SIZE + 2*constants.EMAIL_EMB_SIZE
        inp_dims = 2 * constants.USER_EMB_SIZE + 2 * constants.EMAIL_EMB_SIZE
        self.hidden_layers, out_dim = self.create_hidden_layers(inp_dims, hidden_dims, use_batchnorm)
        # final linear layer that outputs the predicted middle word representation
        self.email_layer = nn.Linear(out_dim, constants.EMAIL_EMB_SIZE)
        if pre_trained:
            self.load(load_from)

    def forward(self, s_id, r_ids, prev_next_embs):
        """
        Input is an integer sender_id, list of receiver_ids and a matrix containing all possible prev_next embeddings.
        The sender and receiver embeddings are extracted using the ids.
        If there are multiple receivers, an average embeddings is taken of all the receivers.
        The sender, receiver embeddings are concatenated with each row of prev_next embs via broadcasting to get one
        matrix, where each row is a sender_emb,recv_emb,prev_emb,next_emb. This matrix is then propagated through the
        layers of neural net
        """
        # convert integers to long tensors
        s_id = autograd.Variable(torch.LongTensor([s_id]))
        r_ids = autograd.Variable(torch.LongTensor(r_ids))
        # extract the embedding for all the receivers and take an average
        r_emb = torch.mean(self.embedding_layer(r_ids), 0, True)
        # extract the sender embedding
        s_emb = self.embedding_layer(s_id)
        # convert prev_next_embs to variable objects
        prev_next_embs = autograd.Variable(torch.from_numpy(prev_next_embs))
        # concatenate the s_emb, r_emb with all the rows in prev_next_emb to get one large matrix
        M = self.concatenate_word_user_emb(s_emb, r_emb, prev_next_embs)
        hidden_out = self.hidden_layers(M)
        word_reps = self.email_layer(hidden_out)
        return word_reps

    def concatenate_word_user_emb(self, s_emb, r_emb, prev_next_embs):
        """
        The method performs the boradcasting action, where it concatenates every row in prev_next_embs with sr_emb
        :param s_emb:
        :param r_emb:
        :param prev_next_embs:
        :return:
        """
        sr_emb = torch.cat([s_emb, r_emb], 1)
        zr = autograd.Variable(torch.zeros(prev_next_embs.shape[0], sr_emb.shape[1]))
        M = torch.cat([(zr + sr_emb), prev_next_embs], 1)
        return M

    def generate_all_combinations(self, email_word_reps):
        """
        The method takes email_word_reps and generates a matrix where each row is w_j-1,w_j+1 and each corresponding
        label is w_j
        """
        curr_embs = []
        prev_next_embs = []
        for i in range(1, len(email_word_reps) - 1):
            curr_embs.append(email_word_reps[i])
            prev_next_embs.append(np.concatenate((email_word_reps[i - 1], email_word_reps[i + 1])))
        return np.array(prev_next_embs), np.array(curr_embs)

    def train(self, emails, val_data, w2v, epochs=10, save_model=True):
        neg_val_data = dal.get_negative_emails(val_data, fraction=1.0)
        optimizer = optim.RMSprop(self.parameters(), lr=1e-3, alpha=0.99, momentum=0.0)

        for epoch in range(epochs):
            epoch_loss = 0.0
            start = time.time()
            # loop over each mail
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
            print 'time taken for epoch:', (end - start)
            print 'loss in epoch ' + str(epoch) + ' = ' + str(epoch_loss)
            self.run_validation(epoch, val_data, neg_val_data, w2v)

        if save_model:
            file_name = constants.RUN_ID + '_finalmodel.pth'
            self.save(file_name)
        email_ids, embs = self.extract_user_embeddings()
        utils.save_user_embeddings(email_ids, embs)
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

        # get word representations from glove word2vec
        email_word_reps = w2v.get_sentence(email_content)
        # generate a matrix that will contain all combinations of w_j-1,w_j+1 - > w_j
        prev_next_embs, curr_embs = self.generate_all_combinations(email_word_reps)
        if len(curr_embs) == 0:
            return 0, False
        # do the forward pass
        pred_word_reps = self.forward(sender_id, recv_ids, prev_next_embs)
        # compute the loss
        loss = loss_criteria(pred_word_reps, autograd.Variable(torch.from_numpy(curr_embs)))
        return loss, True

    def get_repr(self, identifier):
        pass
