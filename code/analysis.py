"""
train, val, test = dal.dataset_filter_by_user(emails, val_split=0.1, test_split=0.2, threshold=0)
num_train, num_val, num_test = train.shape[0], val.shape[0], test.shape[0]
mails_grouped_by_sender = utils.group_mails_by_sender(test)
num_filtered_users = len(mails_grouped_by_sender.keys())
print train.shape, val.shape, test.shape, 'filtered users:', num_filtered_users
for threshold in xrange(5, 50 + 1, 5):
    train, val, test = dal.dataset_filter_by_user(emails, val_split=0.1, test_split=0.2, threshold=threshold)
    num_train_cur, num_val_cur, num_test_cur = train.shape[0], val.shape[0], test.shape[0]
    emails_lost = num_train + num_val + num_test - (num_train_cur + num_val_cur + num_test_cur)
    mails_grouped_by_sender = utils.group_mails_by_sender(test)
    num_filtered_users_cur = len(mails_grouped_by_sender.keys())
    users_lost = num_filtered_users - num_filtered_users_cur
    print 'Threshold: {} Emails lost: {} Users lost: {}'.format(threshold, emails_lost, users_lost)

Number of emails returned by dal 28060
(19579, 4) (2739, 4) (5741, 4) filtered users: 144
Threshold: 5 Emails lost: 37 Users lost: 18
Threshold: 10 Emails lost: 88 Users lost: 30
Threshold: 15 Emails lost: 144 Users lost: 39
Threshold: 20 Emails lost: 178 Users lost: 43
* Threshold: 25 Emails lost: 250 Users lost: 50 *
Threshold: 30 Emails lost: 336 Users lost: 57
Threshold: 35 Emails lost: 439 Users lost: 64
Threshold: 40 Emails lost: 491 Users lost: 67
Threshold: 45 Emails lost: 549 Users lost: 70
Threshold: 50 Emails lost: 570 Users lost: 71
"""

"""
import matplotlib.pyplot as plt
import os

if not os.path.exists('../outputs/{}'.format(constants.RUN_ID)):
    os.mkdir('../outputs/{}'.format(constants.RUN_ID))
    os.mkdir('../outputs/{}/l2-errors'.format(constants.RUN_ID))
elif not os.path.exists('../outputs/{}/l2-errors'.format(constants.RUN_ID)):
    os.mkdir('../outputs/{}/l2-errors'.format(constants.RUN_ID))

train_mails_grouped_by_sender = utils.group_mails_by_sender(train)
val_mails_grouped_by_sender = utils.group_mails_by_sender(val)
test_mails_grouped_by_sender = utils.group_mails_by_sender(test)
for sender in val_mails_grouped_by_sender:
    _, train_errors = metrics_utils.get_predictions(model, w2v, train_mails_grouped_by_sender[sender],
                                                    neg_emails=[], is_l2=True)
    _, val_errors = metrics_utils.get_predictions(model, w2v, val_mails_grouped_by_sender[sender],
                                                  neg_emails=[], is_l2=True)
    _, test_errors = metrics_utils.get_predictions(model, w2v, test_mails_grouped_by_sender[sender],
                                                   neg_emails=[], is_l2=True)
    plt.close()
    x = [train_errors, val_errors, test_errors]
    plt.hist(x, 10, histtype='bar', color=['b', 'g', 'r'])

    plt.title(sender)
    plt.savefig('../outputs/{}/l2-errors/{}.png'.format(constants.RUN_ID, sender))
"""
