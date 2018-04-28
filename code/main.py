"""
This script orchestrates the entire model's lifecycle via a single pipeline
Multiple flags can be used to enable and disable certain sections of the pipeline
"""

import numpy as np
import sys
import time

import dal
import metrics
import utils

from model1 import Model1
from model3 import Model3
from model4 import Model4
from model2faster import Model2Faster
from w2v_custom import W2VCustom
from w2v_glove import W2VGlove
import constants


def get_predictions_l2(model, w2v, emails, neg_emails):
    y_true = []
    y_pred = []
    for i in xrange(len(emails)):
        loss, valid = model.predict(emails[i, :], w2v)
        if valid:
            y_true.append(1)
            y_pred.append(np.asscalar(loss.data.numpy()))

    for i in xrange(len(neg_emails)):
        loss, valid = model.predict(neg_emails[i, :], w2v)
        if valid:
            y_true.append(0)
            y_pred.append(np.asscalar(loss.data.numpy()))

    return np.array(y_true), np.array(y_pred)


def get_predictions_prob(model, w2v, emails, neg_emails):
    y_true = []
    y_pred = []
    for i in xrange(len(emails)):
        prob, valid = model.predict(emails[i, :], w2v)
        if valid:
            y_true.append(1)
            y_pred.append(prob.data.numpy()[0, 1])

    for i in xrange(len(neg_emails)):
        prob, valid = model.predict(neg_emails[i, :], w2v)
        if valid:
            y_true.append(0)
            y_pred.append(prob.data.numpy()[0, 1])

    return np.array(y_true), np.array(y_pred)


def stats(hit_positions, k_values):
    for k in k_values:
        print k, np.sum(hit_positions < k) / (len(hit_positions) + 0.0)


def calc_updated_metrics(model, w2v, emails, k=10):
    employee_list, _ = model.extract_user_embeddings()
    k = min(k, len(employee_list))
    num_hits = 0.0
    num_total = 0
    hit_positions = []
    for i in xrange(emails.shape[0]):
        actual_sender_idx = employee_list.index(emails[i, constants.SENDER_EMAIL])
        this_email = emails[i, :]
        this_y_true = []
        this_y_pred = []
        for employee in employee_list:
            this_email[constants.SENDER_EMAIL] = employee
            loss, valid = model.predict(this_email, w2v)
            if valid:
                num_total += 1 if employee == employee_list[actual_sender_idx] else 0
                this_y_true.append(1 if employee == employee_list[actual_sender_idx] else 0)
                this_y_pred.append(np.asscalar(loss.data.numpy()))
        this_y_true = np.array(this_y_true)
        this_y_pred = np.array(this_y_pred)
        sort_order = this_y_pred.argsort()
        num_hits += np.sum(this_y_true[sort_order][:k])
        if this_y_true.shape[0]:
            hit_positions.append(np.argmax(this_y_true[sort_order]))
    return num_hits, num_total, num_hits / num_total, np.array(hit_positions)


if __name__ == '__main__':
    # TODO: Consider moving parameters and hyperparameters into another file?
    # Or to be injected via the command line
    start = time.time()

    # reading command-line argument
    args = sys.argv
    model_name = args[2]
    num_epochs = int(args[3])
    if len(sys.argv) > 4:
        num_users = int(args[4])
    else:
        num_users = 150
    utils.populate_userid_mapping()
    NUM_EMAILS = 10000

    model = locals()[model_name](pre_trained=True, load_from='{}_model.pth'.format(constants.RUN_ID))
    # w2v = W2VCustom()
    w2v = W2VGlove()

    # emails = dal.get_emails(fetch_all=True)
    emails = dal.get_emails_by_users(num_users=num_users)
    print 'Number of emails returned by dal', len(emails)

    train, val, test = dal.dataset_split(emails, val_split=0.1, test_split=0.2)

    # w2v.train(emails)

    email_body = emails[0][2]
    sentence = w2v.get_sentence(email_body)
    print sentence[0].shape

    # start = time.time()
    # utils.get_nearest_neighbors_emails(emails, w2v, 5)
    # end = time.time()
    # print 'time taken = ', (end-start)

    # model.train(train, w2v, num_epochs)

    neg_emails = dal.get_negative_emails(test, fraction=1.0)
    print 'Number of negative emails returned by dal', len(neg_emails)

    mails_grouped_by_sender = utils.group_mails_by_sender(test)

    if model_name in ('Model1', 'Model2', 'Model2Faster', 'Model3'):
        y_true, y_pred = get_predictions_l2(model, w2v, test, neg_emails)
        print metrics.hits_at_k(y_true, y_pred, k=1000, is_l2=True)
        print metrics.average_precision_at_k(y_true, y_pred, k=1000, is_l2=True)
        # y_true_all, y_pred_all = [], []
        # for sender in mails_grouped_by_sender:
        #     this_y_true, this_y_pred = get_predictions_l2(model, w2v, mails_grouped_by_sender[sender], [])
        #     y_true_all.append(this_y_true)
        #     y_pred_all.append(this_y_pred)
        # print metrics.mean_average_precision_at_k(y_true_all, y_pred_all)
    elif model_name in ('Model4', ):
        y_true, y_pred = get_predictions_prob(model, w2v, test, neg_emails)
        print metrics.hits_at_k(y_true, y_pred, k=1000, is_l2=False)
        print metrics.average_precision_at_k(y_true, y_pred, k=1000, is_l2=False)

    # num_hits, num_total, h_by_t, hit_positions = calc_updated_metrics(model, w2v, test, k=10)
    # print num_hits, num_total, h_by_t
    #
    # print stats(hit_positions, k_values=[5, 10, 20, 30])

    print 'End of script! Time taken ' + str(time.time() - start)
