import numpy as np

import utils

from scipy.stats import norm


def get_predictions(model, w2v, emails, neg_emails, is_l2):
    # If is_l2 is set to False, the output loss is interpreted as a probability
    y_true = []
    y_pred = []
    for i in xrange(len(emails)):
        loss, valid = model.predict(emails[i, :], w2v)
        if valid:
            y_true.append(1)
            if is_l2:
                y_pred.append(np.asscalar(loss.data.numpy()))
            else:
                y_pred.append(loss.data.numpy()[0, 1])

    for i in xrange(len(neg_emails)):
        loss, valid = model.predict(neg_emails[i, :], w2v)
        if valid:
            y_true.append(0)
            if is_l2:
                y_pred.append(np.asscalar(loss.data.numpy()))
            else:
                y_pred.append(loss.data.numpy()[0, 1])

    return np.array(y_true), np.array(y_pred)


def get_error_distributions(model, w2v, emails):
    distributions = {}
    mails_by_sender = utils.group_mails_by_sender(emails)
    for sender, mails in mails_by_sender.items():
        distributions[sender] = {}
        _, errors = get_predictions(model, w2v, mails, neg_emails=[], is_l2=True)
        mu, std = norm.fit(errors)

        distributions[sender]['train_errors'] = errors
        distributions[sender]['mu'] = mu
        distributions[sender]['std'] = std

    return distributions
