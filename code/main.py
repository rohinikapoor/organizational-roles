"""
This script orchestrates the entire model's lifecycle via a single pipeline
Multiple flags can be used to enable and disable certain sections of the pipeline
"""

from model1 import Model1
from model2 import Model2
from model3 import Model3
from w2v_custom import W2VCustom
from w2v_glove import W2VGlove

import utils
import dal
import time


if __name__ == '__main__':
    # TODO: Consider moving parameters and hyperparameters into another file?
    # Or to be injected via the command line
    start = time.time()
    utils.populate_userid_mapping()
    NUM_EMAILS = 1000

    # model = Model1(epochs=50)
    # model = Model2()
    model = Model3(epochs=50)
    # w2v = W2VCustom()
    w2v = W2VGlove()

    # emails = dal.get_emails(fetch_all=True)
    emails = dal.get_emails_by_users()
    print 'Number of emails returned by dal', len(emails)
    w2v.train(emails)

    # email_body = emails[0][2]
    # sentence = w2v.get_sentence(email_body)

    # start = time.time()
    # utils.get_nearest_neighbors_emails(emails, w2v, 5)
    # end = time.time()
    # print 'time taken = ', (end-start)
    model.train(emails, w2v)

    print 'End of script! Time taken ' + str(time.time() - start)
