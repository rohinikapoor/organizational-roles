import numpy as np

import constants
import metrics_utils
import utils


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC


def _separate_pos_neg(y_true, y_pred, is_l2):
    """
    This helper function splits the dataset (y_true) into y_true_pos and y_true_neg which are the classifier's ranked
    outputs to the queries P(valid = 1| sender,receiver,email) and P(valid = 0| sender,receiver,email)
    """
    if is_l2:
        sort_order = y_pred.argsort()
        y_true_pos = y_true[sort_order]
        y_true_neg = np.flip(1 - y_true[sort_order], 0)
    else:
        sort_order = (-y_pred).argsort()
        y_pred = y_pred[sort_order]
        y_true = y_true[sort_order]
        y_true_pos = y_true[y_pred > 0.5]
        y_true_neg = np.flip(1 - y_true[y_pred < 0.5], 0)
    return y_true_pos, y_true_neg


def _get_effective_k(y_true_pos, y_true_neg, k):
    if k is None:
        k_pos = len(y_true_pos)
        k_neg = len(y_true_neg)
    else:
        k_pos = min(k, len(y_true_pos))
        k_neg = min(k, len(y_true_neg))
    return k_pos, k_neg


def hits_at_k(y_true, y_pred=None, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message
    Hits @ k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: ndarray having shape (num_examples, ) indicating whether the mail was sent by the s-r pair
    :param y_pred: ndarray having shape (num_examples, ) containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: (<positive_hits_at_k>, <k_pos>, <negative_hits_at_k>, <k_neg>)
    """
    y_true_pos, y_true_neg = _separate_pos_neg(y_true, y_pred, is_l2)
    k_pos, k_neg = _get_effective_k(y_true_pos, y_true_neg, k)

    hits_pos = np.sum(y_true_pos[:k_pos])
    hits_neg = np.sum(y_true_neg[:k_neg])
    return hits_pos, k_pos, hits_neg, k_neg


def _calc_average_precision(y_true, k):
    correct = 0
    ap = 0.0
    for i in xrange(k):
        correct += y_true[i]
        precision = correct / (i + 1.0)
        ap += precision * y_true[i]
    return 1 if min(k, np.asscalar(np.sum(y_true))) == 0 else ap / min(k, np.asscalar(np.sum(y_true)))


def average_precision_at_k(y_true, y_pred=None, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message
    Average precision at k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: ndarray having shape (num_examples, ) indicating whether the mail was sent by the s-r pair
    :param y_pred: ndarray having shape (num_examples, ) containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: (<positive_ap_at_k>, <k_pos>, <negative_ap_at_k>, <k_neg>)
    """
    y_true_pos, y_true_neg = _separate_pos_neg(y_true, y_pred, is_l2)
    k_pos, k_neg = _get_effective_k(y_true_pos, y_true_neg, k)

    ap_pos = _calc_average_precision(y_true_pos, k_pos)
    ap_neg = _calc_average_precision(y_true_neg, k_neg)
    return ap_pos, k_pos, ap_neg, k_neg


def mean_average_precision_at_k(y_true, y_pred, k=None, is_l2=False):
    """
    Given distances between expected email representation and actual email representation,
    or distribution of the probability that a sender receiver pair has sent a message for Q users
    Mean average precision at k tells us that of the most confident k predictions, how many of them were correct
    :param y_true: iterable of length Q, each element is an ndarray having shape (num_examples, )
                                            indicating whether the mail was sent by the s-r pair
    :param y_pred: iterable of length Q, each element is an ndarray having shape (num_examples, )
                                            containing scores or probabilities
    :param k: custom value for k; if this is not set, it considers the entire array
    :param is_l2: flag to indicate whether the y_pred values are scores/errors (True) or probabilities (False)
    :return: (<positive_map>, <negative_map>) - mean average precision @ k for all Q users
    """
    mean_ap_pos, mean_ap_neg = 0.0, 0.0
    for user_y_true, user_y_pred in zip(y_true, y_pred):
        ap_pos, _, ap_neg, _ = average_precision_at_k(user_y_true, user_y_pred, k, is_l2)
        mean_ap_pos += ap_pos
        mean_ap_neg += ap_neg
    return mean_ap_pos / len(y_true), mean_ap_neg / len(y_true)


def stats(hit_positions, k_values):
    for k in k_values:
        print k, np.sum(hit_positions < k) / (len(hit_positions) + 0.0)


def _rank_senders_by_emails(model, w2v, emails, k=10, is_l2=True):
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
                if is_l2:
                    this_y_pred.append(np.asscalar(loss.data.numpy()))
                else:
                    this_y_pred.append(loss.data.numpy()[0, 1])

        this_y_true = np.array(this_y_true)
        this_y_pred = np.array(this_y_pred)
        if is_l2:
            sort_order = this_y_pred.argsort()
        else:
            sort_order = (-this_y_pred).argsort()
        num_hits += np.sum(this_y_true[sort_order][:k])
        if this_y_true.shape[0]:
            hit_positions.append(np.argmax(this_y_true[sort_order]))
    return num_hits, num_total, num_hits / num_total, np.array(hit_positions)


def evaluate_metrics(model, model_name, w2v, test, neg_emails, k=1000,
                     metrics=('hits@k', 'ap@k', 'map@k', 'ryan-hits@k')):
    """
    :param model: trained model whose performance is to be evaluated
    :param model_name: which class the model belongs to
    :param w2v: word vector embedding model
    :param test: test emails to run the evaluation on
    :param neg_emails: negatively sampled emails to run the evaluation on
    :param k: number of emails to be considered for evaluation
    :param metrics: types of metrics to be calculated
    :return: None
    """
    if model_name in ('Model1', 'Model2', 'Model3'):
        is_l2 = True
    elif model_name in ('Model4', ):
        is_l2 = False
    else:
        raise NotImplementedError('Model {} has not been implemented'.format(model_name))

    y_true, y_pred = metrics_utils.get_predictions(model, w2v, test, neg_emails, is_l2)

    if 'hits@k' in metrics:
        print 'Hits@{}'.format(k), hits_at_k(y_true, y_pred, k=k, is_l2=is_l2)
    if 'ap@k' in metrics:
        print 'AP@{}'.format(k), average_precision_at_k(y_true, y_pred, k=k, is_l2=is_l2)
    if 'map@k' in metrics:
        mails_grouped_by_sender = utils.group_mails_by_sender(test)
        y_true_all, y_pred_all = [], []
        for sender in mails_grouped_by_sender:
            this_y_true, this_y_pred = metrics_utils.get_predictions(model, w2v, mails_grouped_by_sender[sender], [],
                                                                     is_l2)
            y_true_all.append(this_y_true)
            y_pred_all.append(this_y_pred)
        print 'MAP@{}'.format(k), mean_average_precision_at_k(y_true_all, y_pred_all)
    if 'ryan-hits@k' in metrics:
        num_hits, num_total, h_by_t, hit_positions = _rank_senders_by_emails(model, w2v, test, k=10, is_l2=is_l2)
        print 'Rank senders by emails:', stats(hit_positions, k_values=[5, 10, 20, 30])


def k_fold_cross_validation(email_ids, embs):
    # extract the data
    X, y = utils.extract_emb_desgn(email_ids, embs, cat='cat1')
    print np.unique(y, return_counts=True)
    # split the data into k-folds
    kf = KFold(n_splits=23, shuffle=True)
    # run k-fold cross validation
    cor = 0
    train_acc = []
    y_t = np.array([])
    y_p = np.array([])
    correct_class = {}
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        classifier = SVC(C=1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                c_score = correct_class.get(y_test[i], 0)
                c_score += 1
                correct_class[y_test[i]] = c_score

        y_train_pred = classifier.predict(X_train)
        cor = cor + np.sum(y_pred == y_test)
        train_acc.append((1.0*np.sum(y_train_pred == y_train))/len(y_train))
        y_t = np.append(y_t, y_test)
        y_p = np.append(y_p, y_pred)
    print 'test acc:', (cor * 1.0) / len(y)
    print 'train acc:', np.mean(train_acc)
    print correct_class
    print np.unique(y, return_counts=True)

    #plot the confusion matrix
    print confusion_matrix(y_t, y_p)
    print classification_report(y_t, y_p)


def dominance_metric(email_ids, embs):
    train_users, train_embs, test_users, test_embs = utils.split_by_users(email_ids, embs)
    # X, y = utils.get_dominance_data(email_ids, embs)
    # print np.unique(y, return_counts=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    X_train, y_train = utils.get_dominance_data(train_users, train_embs)
    print np.unique(y_train, return_counts=True)
    X_test, y_test = utils.get_dominance_data(test_users, test_embs)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape
    classifier = SVC(C=0.1)
    classifier.fit(X_train, y_train)
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    print 'train_acc:', np.mean(y_pred_train == y_train)
    print 'test_acc:', np.mean(y_pred_test == y_test)
    print np.unique(y_test, return_counts=True)
    print confusion_matrix(y_test, y_pred_test)
    print classification_report(y_test, y_pred_test)


if __name__ == '__main__':
    # email_ids, embs = utils.load_user_embeddings(
    #     '../important_embeddings/usr100d_em200d_20ep_m3/embeddings_usr100d_em200d_20ep_m3.pkl')
    # k_fold_cross_validation(email_ids, embs)
    email_ids, embs = utils.load_user_embeddings('../resources/embeddings_dundundun.pkl')
    dominance_metric(email_ids, embs)
    # y_true = np.array([1, 0, 1, 0, 1])
    # # expected sort order -> 1 1 0 0 1
    # y_pred_prob = np.array([0.6, 0.4, 0.7, 0.55, 0.0])
    # y_pred_l2 = np.array([10, 30, 20, 40, 50])
    #
    # # 3 on 5
    # print hits_at_k(y_true, y_pred_l2, is_l2=True)
    # print hits_at_k(y_true, y_pred_prob, is_l2=False)
    #
    # # (1/1 + 2/2 + 3/5) / 3
    # print average_precision_at_k(y_true, y_pred_l2, is_l2=True)
    # print average_precision_at_k(y_true, y_pred_prob, is_l2=False)
    #
    # y2_true = np.array([1, 1, 0, 1, 1])
    # # expected sort order -> 1 0 1 1 1
    # y2_pred_prob = np.array([0.7, 0.4, 0.6, 0.55, 0.0])
    # y2_pred_l2 = np.array([10, 30, 20, 40, 50])
    #
    # # (1/1 + 2/3 + 3/4 + 4/5) / 4
    # print average_precision_at_k(y2_true, y2_pred_l2, is_l2=True)
    # print average_precision_at_k(y2_true, y2_pred_prob, is_l2=False)
    #
    # # (0.866 + 0.804) / 2
    # print mean_average_precision_at_k([y_true, y2_true], [y_pred_l2, y2_pred_l2], is_l2=True)
    # print mean_average_precision_at_k([y_true, y2_true], [y_pred_prob, y2_pred_prob], is_l2=False)
