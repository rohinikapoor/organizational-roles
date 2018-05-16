"""
This script orchestrates the entire model's lifecycle via a single pipeline
Multiple flags can be used to enable and disable certain sections of the pipeline
"""

import numpy as np
import pickle
import sys
import time

import constants
import dal
import metrics
import metrics_utils
import plots
import utils

from model1 import Model1
from model3 import Model3
from model4 import Model4
from model2 import Model2
from model2deeper import Model2Deeper
from model3deeper import Model3Deeper
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
    PRE_TRAINED = True

    model = locals()[model_name](pre_trained=PRE_TRAINED, load_from='usr50d_em50d_25ep_Model2_256_256_256_model.pth',
                                 hidden_dims=constants.HIDDEN_DIMS)
    # w2v = W2VCustom()
    w2v = W2VGlove()

    # emails = dal.get_emails(fetch_all=True)
    emails = dal.get_emails_by_users(num_users=num_users)
    print 'Number of emails returned by dal', len(emails)

    # train, val, test = dal.dataset_split(emails, val_split=0.1, test_split=0.2)
    train, val, test = dal.dataset_filter_by_user(emails, val_split=0.1, test_split=0.2, threshold=25)

    # w2v.train(emails)

    # email_body = emails[0][2]
    # sentence = w2v.get_sentence(email_body)
    # print sentence[0].shape

    # start = time.time()
    # utils.get_nearest_neighbors_emails(emails, w2v, 5)
    # end = time.time()
    # print 'time taken = ', (end-start)

    if not PRE_TRAINED:
        model.train(train, val, w2v, num_epochs)

    if PRE_TRAINED:
        distributions = pickle.load(open('../outputs/{}-distributions.pkl'.format(constants.RUN_ID), 'rb'))
    else:
        distributions = metrics_utils.get_error_distributions(model, w2v, train)
        tr = utils.group_mails_by_sender(train)
        va = utils.group_mails_by_sender(val)
        te = utils.group_mails_by_sender(test)
        names = [x for x in te if x in va and x in tr]
        print names

        for sender in distributions:
            if sender not in names:
                distributions[sender] = None
            else:
                _, val_errors = metrics_utils.get_predictions(model, w2v, va[sender], neg_emails=[], is_l2=True)
                _, test_errors = metrics_utils.get_predictions(model, w2v, te[sender], neg_emails=[], is_l2=True)
                distributions[sender]['val_errors'] = val_errors
                distributions[sender]['test_errors'] = val_errors
        distributions = {key: value for key, value in distributions.items() if not value is None}
        pickle.dump(distributions, open('../outputs/{}-distributions.pkl'.format(constants.RUN_ID), 'wb'))

    # metrics.test_error_deviation_thresholds(distributions)

    # This will be a little different when compared to the other mails
    # The last column is describes the type of email instead of containing the date
    special_emails = dal.get_emails_for_anomaly_testing()
    special_emails_by_sender = utils.group_mails_by_sender(special_emails)

    # for sender in ['kimberly.watson@enron.com']:
    for sender in ['carol.clair@enron.com', 'lynn.blair@enron.com']:
        # plots.plot_error_distribution_v2(model, w2v, sender, tr[sender], va[sender], te[sender])
        plots.plot_error_distribution(sender, distributions[sender]['train_errors'],
                                      distributions[sender]['val_errors'], distributions[sender]['test_errors'])

        _, special_errors = metrics_utils.get_predictions(model, w2v, special_emails_by_sender[sender],
                                                          neg_emails=[], is_l2=True)
        plots.plot_special_mails(sender, distributions, special_emails_by_sender[sender], special_errors)

    # neg_emails = dal.get_negative_emails(test, fraction=1.0)
    # print 'Number of negative emails returned by dal', len(neg_emails)

    # metrics.evaluate_metrics(model, model_name, w2v, test, neg_emails, k=1000,
    #                          metrics=['hits@k', 'ap@k', 'map@k', 'ryan-hits@k'])
    # metrics.evaluate_metrics(model, model_name, w2v, test, neg_emails, k=1000, metrics=['hits@k', 'ap@k'])

    print 'End of script! Time taken ' + str(time.time() - start)
