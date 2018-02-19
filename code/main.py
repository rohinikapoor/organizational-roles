"""
This script orchestrates the entire model's lifecycle via a single pipeline
Multiple flags can be used to enable and disable certain sections of the pipeline
"""

from model1 import Model1
from model2 import Model2
from w2v_custom import W2VCustom
from w2v_glove import W2VGlove

import utils
import dal


if __name__ == '__main__':
    # TODO: Consider moving parameters and hyperparameters into another file?
    # Or to be injected via the command line
    NUM_EMAILS = 1000

    model = Model1()
    # model = Model2()
    w2v = W2VCustom()
    w2v = W2VGlove()

    emails = dal.get_emails(NUM_EMAILS)
    w2v.train(emails)

    email_body = emails[0][2]
    sentence = w2v.get_sentence(email_body)
    print len(sentence), email_body

    model.train(emails, w2v)

    print 'End of script!'
