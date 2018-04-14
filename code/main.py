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
from model2 import Model2
from model3 import Model3
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

    model = locals()[model_name](epochs=num_epochs)
    # w2v = W2VCustom()
    w2v = W2VGlove()

    # emails = dal.get_emails(fetch_all=True)
    emails = dal.get_emails_by_users(num_users=num_users)
    print 'Number of emails returned by dal', len(emails)
    w2v.train(emails)

    email_body = emails[0][2]
    sentence = w2v.get_sentence(email_body)
    print sentence[0].shape

    # start = time.time()
    # utils.get_nearest_neighbors_emails(emails, w2v, 5)
    # end = time.time()
    # print 'time taken = ', (end-start)
    # model.train(emails, w2v)

    neg_emails = dal.get_negative_emails(emails, fraction=1.0)
    print 'Number of negative emails returned by dal', len(neg_emails)

    if model_name in ('Model1', 'Model2', 'Model2Faster', 'Model3'):
        y_true, y_pred = get_predictions_l2(model, w2v, emails, neg_emails)
        print metrics.hits_at_k(y_true, y_pred, k=1000, is_l2=True)
        print metrics.average_precision_at_k(y_true, y_pred, k=1000, is_l2=True)

    print 'End of script! Time taken ' + str(time.time() - start)
