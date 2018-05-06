"""
This script orchestrates the entire model's lifecycle via a single pipeline
Multiple flags can be used to enable and disable certain sections of the pipeline
"""

import sys
import time

import constants
import dal
import metrics
import utils

from model1 import Model1
from model3 import Model3
from model4 import Model4
from model2faster import Model2Faster
from w2v_custom import W2VCustom
from w2v_glove import W2VGlove


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
    PRE_TRAINED = False

    model = locals()[model_name](pre_trained=PRE_TRAINED, load_from='{}_model.pth'.format(constants.RUN_ID))
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

    if not PRE_TRAINED:
        model.train(train, w2v, num_epochs)

    neg_emails = dal.get_negative_emails(test, fraction=1.0)
    print 'Number of negative emails returned by dal', len(neg_emails)

    # metrics.evaluate_metrics(model, model_name, w2v, test, neg_emails, k=1000,
    #                          metrics=['hits@k', 'ap@k', 'map@k', 'ryan-hits@k'])
    metrics.evaluate_metrics(model, model_name, w2v, test, neg_emails, k=1000, metrics=['hits@k', 'ap@k'])

    print 'End of script! Time taken ' + str(time.time() - start)
